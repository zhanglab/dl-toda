import sys
import os
import glob
import argparse
import math
import zipfile
import subprocess
import multiprocessing
import random
import statistics
import numpy as np
import json
from Bio import SeqIO, SeqUtils
from collections import defaultdict
from pycirclize import Circos, config
from Bio.SeqFeature import SeqFeature, FeatureLocation
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from dataprep_scripts.utils import load_fq_file
# from genomics_viz_utils import *
from vis_scripts.parse_samfile import LoadData, GetCoverageOfSample
from pygenomeviz.parser import Fasta
from pygenomeviz.utils import load_example_fasta_dataset, ColorCycler, interpolate_color
from pygenomeviz.align import AlignCoord, Blast
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

ColorCycler.set_cmap("Set1")

# QUERY_TRACK_SIZE = 5
MIN_IDENTITY = 70
TICKS_INTERVAL = 500000
bowtie2_build_exec = "/modules/uri_apps/software/Bowtie2/2.4.5-GCC-11.3.0/bin/bowtie2-build"
bowtie2_exec = "/modules/uri_apps/software/Bowtie2/2.4.5-GCC-11.3.0/bin/bowtie2"
blastn_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/blastn"
makeblastdb_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/makeblastdb"
ncbi_datasets_exec = "/work/pi_yingzhang_uri_edu/ccres/tools/datasets"
parallel_exec = "/modules/spack/packages/linux-ubuntu24.04-x86_64_v3/gcc-13.2.0/parallel-20240822-uwvjfxdji5ltqgl6vdu4in522ymhbhz7/bin/parallel"
# set seed
seed = 42
# set the global python random seed
random.seed(seed)


def GetLocusSeq(start, end, strand, training_seq):
	if end < start:
		seq = training_seq[end:start+1]
	elif end > start:
		seq = training_seq[start:end+1]
	
	return seq

def GetGIAlignments(input_file):
	alignments = defaultdict(list)
	with open(input_file, 'r') as f:
		for line in f:
			gi_id = line.rstrip().split(',')[0]
			sstart = int(line.rstrip().split(',')[2])
			send = int(line.rstrip().split(',')[3])
			evalue = float(line.rstrip().split(',')[7])
			pident = float(line.rstrip().split(',')[8])
			sstrand = line.rstrip().split(',')[11]
			if gi_id in alignments:
				if evalue < alignments[gi_id][2] and pident > alignments[gi_id][3]:
					alignments[gi_id] = [sstart, send, evalue, pident, sstrand]
			else:
				alignments[gi_id] = [sstart, send, evalue, pident, sstrand]

	return alignments

def GetGIsFromAnnotations(args, input_dir, sequence, genome_id, fasta):
	outf = open(os.path.join(args.output_dir, f'{args.label}_{genome_id}_gis.tsv'), 'w')
	fna = open(os.path.join(args.output_dir, f'{args.label}_{genome_id}_genomic_islands.fna'), 'w')

	# get gene annotations 
	pos_train_annot_info, locus_tags_info = GetAnnotInfo(args, genome_id, input_dir)
	# parse annotation information
	annot_parsed = defaultdict(list)
	for gene_id, data in locus_tags_info.items():
		if data[2] != '':
			annot_parsed[data[2]] = [data[0], data[1], gene_id, data[3]]
	
	gis_info = defaultdict(list)
	with open(args.genomic_islands, 'r') as f:
		for line in f:
			gi_id = line.rstrip().split('\t')[1]
			start_locus_tag = line.rstrip().split('\t')[2]
			end_locus_tag = line.rstrip().split('\t')[3]
			if start_locus_tag in annot_parsed and end_locus_tag in annot_parsed:
				start_locus_tag_start = annot_parsed[start_locus_tag][0]
				start_locus_tag_end = annot_parsed[start_locus_tag][1]
				start_new_locus_tag = annot_parsed[start_locus_tag][2]
				start_locus_strand = annot_parsed[start_locus_tag][3]

				end_locus_tag_start = annot_parsed[end_locus_tag][0]
				end_locus_tag_end = annot_parsed[end_locus_tag][1]
				end_new_locus_tag = annot_parsed[end_locus_tag][2]
				end_locus_strand = annot_parsed[end_locus_tag][3]

				gis_info[f'{gi_id}_start'] = [start_locus_tag_start, start_locus_tag_end, start_locus_strand]
				gis_info[f'{gi_id}_end'] = [end_locus_tag_start, end_locus_tag_end, end_locus_strand]

				start_locus_sequence = GetLocusSeq(start_locus_tag_start, start_locus_tag_end, start_locus_strand, sequence)
				end_locus_sequence = GetLocusSeq(end_locus_tag_start, end_locus_tag_end, end_locus_strand, sequence)

				fna.write(f'>{gi_id}_start\n{start_locus_sequence}\n')
				fna.write(f'>{gi_id}_end\n{end_locus_sequence}\n')

				outf.write(f'{gi_id}\t{start_locus_tag}\t{start_new_locus_tag}\t{start_locus_tag_start}\t{start_locus_tag_end}\t{start_locus_strand}\t{end_locus_tag}\t{end_new_locus_tag}\t{end_locus_tag_start}\t{end_locus_tag_end}\t{end_locus_strand}\n')

	outf.close()
	fna.close()

	# blast GIs start and end loci to testing genome
	RunBlast(args, os.path.join(args.output_dir, 'blast', f'gis_{genome_id}_genome'), os.path.join(args.output_dir, f'{args.label}_{genome_id}_genomic_islands.fna'), subject=[fasta], outfilename=f'{args.output_dir}/blast/gis_{genome_id}_genome/gis_blastn.out')
	gis_align = GetGIAlignments(f'{args.output_dir}/blast/gis_{genome_id}_genome/gis_blastn.out')

	return gis_align


def GetGIsFromFasta(args, genome_id, ref_fasta):
	fna = open(os.path.join(args.output_dir, f'{args.label}_{genome_id}_genomic_islands.fna'), 'w')
	fasta_files = glob.glob(os.path.join(args.genomic_islands, '*.fna'))
	info_file = glob.glob(os.path.join(args.genomic_islands, '*.tsv'))[0]
	for fasta in fasta_files:
		with open(fasta, 'r') as f:
			fna.write(f.read())
	fna.close()

	# blast GIs start and end loci to testing genome
	RunBlast(args, os.path.join(args.output_dir, 'blast', f'gis_{genome_id}_genome'), os.path.join(args.output_dir, f'{args.label}_{genome_id}_genomic_islands.fna'), subject=[ref_fasta], outfilename=f'{args.output_dir}/blast/gis_{genome_id}_genome/gis_blastn.out')
	gis_align = GetGIAlignments(f'{args.output_dir}/blast/gis_{genome_id}_genome/gis_blastn.out')
	# update GIs ID if the information provided consists of the junction sites and not the entire island
	outf = open(os.path.join(args.output_dir, f'{args.label}_{genome_id}_gis.tsv'), 'w')
	with open(info_file, 'r') as f:
		for line in f:
			gi_id = line.rstrip().split('\t')[1]
			pos_info = line.rstrip().split('\t')[2:]
			outf.write(f'{gi_id}\t')
			if len(pos_info) == 2:
				start_locus = pos_info[0]
				end_locus = pos_info[1]
				if start_locus in gis_align:
					start_pos = gis_align[start_locus][0]
					outf.write(f'{start_locus}')
					for e in gis_align[start_locus]:
						outf.write(f'\t{e}')
					
				if end_locus in gis_align:
					end_pos = gis_align[end_locus][1]
					outf.write(f'\t{end_locus}')
					for e in gis_align[end_locus]:
						outf.write(f'\t{e}')
				
				gis_align[gi_id] = [start_pos, end_pos]

				del gis_align[start_locus]
				del gis_align[end_locus]
			else:
				outf.write(f'{pos_info[0]}')
				for e in gis_align[pos_info[0]]:
					outf.write(f'\t{e}')
				gis_align[gi_id] = [gis_align[pos_info[0]][0], gis_align[pos_info[0]][1]]
				del gis_align[pos_info[0]]

			outf.write('\n')
	outf.close()

	return gis_align


def CheckGenomes(args, label):
	# load testing fasta file
	with open(args.test_genomes_info[label][1], "r") as handle:
		test_records = list(SeqIO.parse(handle, "fasta"))

	# load training fasta file
	with open(args.train_genomes_info[label][1], "r") as handle:
		train_records = list(SeqIO.parse(handle, "fasta"))

	assert len(test_records) == 1, f'{label}\t{args.test_genomes_info[label][0]} has more than 1 chromosome'
	assert len(train_records) == 1, f'{label}\t{args.train_genomes_info[label][0]} has more than 1 chromosome'

	return args.test_genomes_info[label][1], test_records, args.train_genomes_info[label][1], train_records


def GetTrainCoverage(args, training_fasta, fn_sequences, tp_sequences, test_alignments_pos_train):
	# get reads in training set fasta file
	readid_to_read, readsid_to_length, _ = LoadFnaFile(args.training_fna_file)

	train_reads_id = []
	with open(os.path.join(args.output_dir, 'train_coverage', f'{args.label}_train_pos_reads.fq'), 'w') as outf:
		for k, v in readid_to_read.items():
			if k.split('|')[1] == args.label:
				outf.write(f'@{k}\n{v}\n+\n{len(v)*"J"}\n')
				train_reads_id.append(k)

	RunBowtie(args, training_fasta, os.path.join(args.output_dir, 'train_coverage', f'{args.label}_train_pos_reads.fq'), os.path.join(args.output_dir, 'train_coverage', f'{args.label}_pos_train_coverage.sam'))
	
	ref_info, alignments = LoadData(os.path.join(args.output_dir, 'train_coverage', f'{args.label}_pos_train_coverage.sam'))
	print(ref_info)

	ref = ref_info[0][0]
	length_ref = ref_info[0][1]
	dict_coverage, reads_info = GetCoverageOfSample(alignments[ref], length_ref, label=None)

	# get coverage per base
	base_coverage = [dict_coverage[i] for i in range(length_ref)]
	total_bases = sum(base_coverage)
	coverage = round(total_bases / length_ref, 3)

	with open(os.path.join(args.output_dir, 'train_coverage', f'{args.label}_{ref}_train_coverage.tsv'), 'w') as f:
		f.write(f'{total_bases}\t{length_ref}\t{coverage}')

	# get train coverage in position mapped by TP reads
	tp_cov = []
	fn_cov = []
	for read_id, data in test_alignments_pos_train.items():
		base_read_cov = [base_coverage[pos-1] for pos in range(data[2], data[3]+1, 1)]
		ave_read_cov = round(sum(base_read_cov)/(data[3]-data[2]), 3)
		if read_id in tp_sequences:
			tp_cov.append(ave_read_cov)
		elif read_id in fn_sequences:
			fn_cov.append(ave_read_cov)

	with open(os.path.join(args.output_dir, f'{args.label}_test_train_average_coverage.tsv'), 'w') as f:
		f.write(f'TP\t{len(tp_cov)}\t{len(tp_sequences)}\t{round(len(tp_cov)/len(tp_sequences),3)*100}\t{statistics.mean(tp_cov)}\t{statistics.median(tp_cov)}\t{min(tp_cov)}\t{max(tp_cov)}\n')
		f.write(f'FN\t{len(fn_cov)}\t{len(fn_sequences)}\t{round(len(fn_cov)/len(fn_sequences),3)*100}\t{statistics.mean(fn_cov)}\t{statistics.median(fn_cov)}\t{min(fn_cov)}\t{max(fn_cov)}\n')

	return base_coverage, ref_info, train_reads_id, readsid_to_length
	

def LoadFnaFile(fasta_file):
	with open(fasta_file, 'r') as f:
		content = f.readlines()
	ordered_reads_id = [content[i].rstrip()[1:] for i in range(0,len(content),2)]
	readsid_to_seq = dict(zip([content[i].rstrip()[1:] for i in range(0, len(content), 2)], [content[i].rstrip() for i in range(1, len(content), 2)]))
	readsid_to_length = dict(zip([content[i].rstrip()[1:] for i in range(0, len(content), 2)], [len(content[i].rstrip()) for i in range(1, len(content), 2)]))

	return readsid_to_seq, readsid_to_length, ordered_reads_id


def RunBowtie(args, target, query, outfilename):
	# build index
	process = subprocess.run([bowtie2_build_exec, '--quiet', '--threads', f'{args.num_processes}', f'{target}', f'{args.output_dir}/train_coverage/ref'])
	# map reads
	process = subprocess.run([bowtie2_exec, '--quiet', '--threads', f'{args.num_processes}', '-x', f'{args.output_dir}/train_coverage/ref', '-U', f'{query}', '-S', f'{outfilename}'])


def RunBlast(args, output_dir, query, subject=None, db=False, outfilename=None, sam=False):
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
	if db:
		sys.executable = blastn_exec
		process = subprocess.run([sys.executable, '-query', f'{query}', '-db', '/datasets/bio/ncbi-db/2025-01-26/nt', '-out', \
			f'{args.output_dir}/blast/test_fp_blastn.out', '-outfmt', "10 delim=, qseqid sseqid evalue pident sstart send qstart qend length ssciname stitle", \
			'-max_target_seqs', '1', '-num_threads', f'{args.num_processes}'])
	else:
		if len(subject) > 1:
			# put all training genomes into one fasta file
			if not os.path.exists(os.path.join(output_dir, 'all_training_genomes.fna')):
				with open(os.path.join(output_dir, 'all_training_genomes.fna'), 'w') as outf:
					for count, fasta in enumerate(subject, 1):
						print(f'{count}\t{fasta}')
						with open(fasta, 'r') as inf:
							outf.write(inf.read())
			input_fasta = os.path.join(output_dir, 'all_training_genomes.fna')
		else:
			input_fasta = subject[0]

		print(input_fasta)
		# create database
		result = subprocess.run([makeblastdb_exec, '-in', f'{input_fasta}', '-input_type', 'fasta', '-dbtype', 'nucl', '-out', f'{output_dir}/blastdb'])
		
		# align reads to database or fasta file
		if sam:
			result = subprocess.run([blastn_exec, '-query', f'{query}', '-db', f'{output_dir}/blastdb', '-out', f'{outfilename}', \
			 	'-outfmt', "17", '-max_target_seqs', '1', '-num_threads', f'{args.num_processes}'])
		else:
			result = subprocess.run([blastn_exec, '-query', f'{query}', '-db', f'{output_dir}/blastdb', '-out', f'{outfilename}', \
			 '-outfmt', "10 delim=, qseqid sseqid sstart send qstart qend qlen evalue pident qseq sseq sstrand", \
			 '-max_target_seqs', '5', '-num_threads', f'{args.num_processes}'])


def GetReadsAlignments(sequences, input_file, sequence_length, seq_to_labels, outfilename=None):
	alignments = defaultdict(list)
	with open(input_file, 'r') as f:
		for line in f:
			readid = line.rstrip().split(',')[0]
			if readid in sequences:
				sstart = int(line.rstrip().split(',')[2])
				send = int(line.rstrip().split(',')[3])
				seq_id = line.rstrip().split(',')[1]
				seq_label = seq_to_labels[seq_id]
				evalue = float(line.rstrip().split(',')[7])
				pident = float(line.rstrip().split(',')[8])
				if readid in alignments:
					if evalue < alignments[readid][4] and pident > alignments[readid][5]:
						alignments[readid] = [seq_label, seq_id, sstart, send, evalue, pident]
				else:
					alignments[readid] = [seq_label, seq_id, sstart, send, evalue, pident]

	if outfilename:
		with open(outfilename, 'w') as f:
			unmapped_reads_id = list(sequences.difference(set(alignments.keys())))
			if len(unmapped_reads_id) != 0:
				unmapped_reads_length = [sequence_length[r] for r in unmapped_reads_id]
				f.write(f'# unmapped reads:\t{len(unmapped_reads_id)}\nmean reads length:\t{statistics.mean(unmapped_reads_length)}\nmedian reads length:\t{statistics.median(unmapped_reads_length)}\nmin reads length:\t{min(unmapped_reads_length)}\nmax reads length:\t{max(unmapped_reads_length)}')
			else:
				f.write(f'# unmapped reads:\t{len(unmapped_reads_id)}\nmean reads length:\tNA\nmedian reads length:\tNA\nmin reads length:\tNA\nmax reads length:\tNA')

	return alignments

def GetMatchRegions(args, input_file, genomic_islands, identity_thr=MIN_IDENTITY):

	align_coords = []
	with open(input_file, 'r') as f:
		for count, line in enumerate(f, 1):
			sstart = int(line.rstrip().split(',')[2])
			send = int(line.rstrip().split(',')[3])
			qstart = int(line.rstrip().split(',')[4])
			qend = int(line.rstrip().split(',')[5])
			pident = float(line.rstrip().split(',')[8])
			qseq = line.rstrip().split(',')[9]
			sseq = line.rstrip().split(',')[10]

			if pident >= identity_thr:
				align_coords.append([qstart, qend, pident])

	return align_coords


def GetGenomesInfo(fasta):
	with open(fasta, 'r') as f:
		content = f.readline()
	strain = ' '.join(content.split(',')[0].split(' ')[1:])
	return strain


def CircosPlot(args, testing_fasta, training_fasta, alignments_train_pos_test, outfigpath, genomic_islands=None):
	
	# load data from training and testing genomes of label 1
	query_fasta = Fasta(training_fasta) # query --> training genome
	ref_fasta_list = list(map(Fasta, testing_fasta)) # ref/subject --> list of testing genomes
	print(testing_fasta)
	print(ref_fasta_list)

	# Initialize circos instance
	circos = Circos(
	    sectors=query_fasta.get_seqid2size(),
	    # space=0 if len(ref_fasta.get_seqid2size()) == 1 else 2,
		space=10,
	)

	circos.text(f'{GetGenomesInfo(training_fasta)}\n(training genome)\n', size=12)
	# get strains of testing genomes
	genomes_id = ['_'.join(i.split('/')[-1].split('_')[2:4]) for i in testing_fasta]
	testing_strains = [GetGenomesInfo(i) for i in testing_fasta]
	print(genomes_id)
	print(testing_strains)

	with open(os.path.join(args.output_dir, f'{args.label}_genomes_length.tsv'), 'w') as f:
		f.write(f'Training genome:\t{query_fasta.name}\t{query_fasta.full_genome_length}\n')
		for ref_fasta in ref_fasta_list:
			f.write(f'Testing genome:\t{ref_fasta.name}\t{ref_fasta.full_genome_length}\n')

	min_r_pos = 100
	for sector in circos.sectors:
		# Plot labels of genomic islands
		if genomic_islands:
			print(genomic_islands)
			color = 'red'
			# add track for genomic islands
			gis_track = sector.add_track((min_r_pos-4, min_r_pos), r_pad_ratio=0.1)
			# f_gis_track = sector.add_track((min_r_pos-3, min_r_pos), r_pad_ratio=0.1)
			# r_gis_track = sector.add_track((min_r_pos-3, min_r_pos), r_pad_ratio=0.1)
			min_r_pos -= 5
			for gi_id in genomic_islands.keys():
				start_locus = genomic_islands[gi_id][0]
				end_locus = genomic_islands[gi_id][1]
				if start_locus > end_locus:
					start_gi = end_locus
					end_gi = start_locus
				else:
					start_gi = start_locus
					end_gi = end_locus

				gis_track.rect(start_gi, end_gi, color=color)
				label_pos = (start_gi + end_gi) / 2
				gis_track.annotate(label_pos, f'{gi_id}', label_size=9)
			print(f'added GIs track')

		# Setup outer track
		outer_track = sector.add_track((min_r_pos-0.3, min_r_pos))
		outer_track.axis(fc="black")
		outer_track.xticks_by_interval(TICKS_INTERVAL, label_formatter=lambda v: f"{v/1000000:.1f} Mb", outer=False,)
		outer_track.xticks_by_interval(100000, tick_length=1, show_label=False)
		min_r_pos -= 6

	
	# store percentage identity between matching regions
	percent_identity = []
	comp_name2color = {}
	train_genome_id = '_'.join(training_fasta[idx].split('/')[-1].split('_')[2:4])
	for idx, ref_fasta in enumerate(ref_fasta_list):
		genome_id = '_'.join(testing_fasta[idx].split('/')[-1].split('_')[2:4])
		RunBlast(args, os.path.join(args.output_dir, 'blast', 'test_train_genomes'), testing_fasta[idx], subject=[training_fasta], outfilename=f'{args.output_dir}/blast/test_train_genomes/{genome_id}_test_train_genomes_blastn.out')
		align_coords = GetMatchRegions(args, f'{args.output_dir}/blast/test_train_genomes/{genome_id}_test_train_genomes_blastn.out', genomic_islands, identity_thr=MIN_IDENTITY)
		color = ColorCycler()
		comp_name2color[genome_id] = color
		for sector in circos.sectors:
			sector.add_track((min_r_pos-5, min_r_pos), r_pad_ratio=0.1)
		for ac in align_coords:
			percent_identity.append(ac[2])
			rect_color = interpolate_color(color, v=ac[2], vmin=MIN_IDENTITY)
			blast_track.rect(ac[0], ac[1], color=rect_color)
		min_r_pos -= 5

		# get stats on percentage identity
		with open(os.path.join(args.output_dir, f'{args.pos_label}_{train_genome_id}_{genome_id}_pct_identity_matching_regions.tsv'), 'w') as f:
			f.write(f'{statistics.mean(percent_identity)}\t{statistics.median(percent_identity)}\t{min(percent_identity)}\t{max(percent_identity)}')
		print(f'{statistics.mean(percent_identity)}\t{statistics.median(percent_identity)}\t{min(percent_identity)}\t{max(percent_identity)}')
	
	# add tracks for coverage of training genome
	for sector in circos.sectors:
		# define x-axis vector for the next track
		genome_pos = list(range(query_fasta.full_genome_length))
		min_r_pos -= 5
		train_cov_train = [0]*query_fasta.full_genome_length
		for readid, data in alignments_train_pos_test.items():
			for pos in range(data[2], data[3]+1, 1):
				train_cov_test[pos-1] += train_coverage[pos-1]
		cov_track = sector.add_track((min_r_pos-10, min_r_pos), r_pad_ratio=0.1)
		cov_track.axis(ec="darkorange")
		y_values = list(range(min(train_coverage), max(train_coverage), 2))
		y_labels = list(map(str, y_values))
		cov_track.yticks(y_values, y_labels)
		cov_track.line(genome_pos, train_cov_test, color="darkorange")
		print(f'added COV track')



	# Save figure
	# Enable annotation text adjustment (Default)
	# config.ann_adjust.enable = True
	fig = circos.plotfig()
	handles = [
		Patch(color='red', label='Pathogenicity Islands'),
		Patch(color='black', label=f'{train_strain}\n(training genome)'),
		]
	handles += [Patch(label=testing_strains[idx], fc=comp_name2color[genomes_id[idx]]) for idx in range(len(testing_fasta))]
	handles += [Patch(color='darkorange', label='Coverage of training\ngenome')]

	_ = circos.ax.legend(handles=handles, bbox_to_anchor=(0.5, 0.475), loc="center", fontsize=8)

	fig.savefig(outfigpath, dpi=300)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--training_fasta', type=str, help='path to file containing list of fasta files')
	parser.add_argument('--testing_fasta', type=str, help='path to file containing path to fasta files of training genomes')
	parser.add_argument('--neg_label', nargs='+', help='list of labels to analyze', required=True)
	parser.add_argument('--pos_label', type=str, help='positive label', required=True)
	parser.add_argument('--testing_fna_file', type=str, help='path to fasta file containing all testing reads (label 1 and 0)')
	parser.add_argument('--training_fna_file', type=str, help='path to fasta file containing all training reads (label 1 and 0)')
	parser.add_argument('--sequences_info', type=str, help='path to file mapping labels of species in model to sequences id of all sequences in training set')
	parser.add_argument('--prob_threshold', type=float, help='probability score threshold', required=True)
	parser.add_argument('--testing_results', type=str, help='path to file containing testing results')
	parser.add_argument('--rank', type=str, help='taxonomic rank investigated', choices=['species','genus','family','order','class', 'phylum'])
	parser.add_argument('--genomic_islands', type=str, help='path to file containing list of genomic islands')
	parser.add_argument('--num_processes', type=int, help='number of processes to run in parallel')
	args = parser.parse_args()
	print(args)

	input_dir = os.getcwd()
	
	# get dltoda taxonomy
	path_dl_toda_tax = '/'.join(
                os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/data/dl_toda_taxonomy.tsv'
	with open(path_dl_toda_tax, 'r') as in_f:
		content = in_f.readlines()
		args.dl_toda_tax = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[1] for line in content}

	# retrieve accession and fasta files of testing and training genomes associated with each label
	with open(args.testing_fasta, 'r') as f:
		content = f.readlines()
		args.test_genomes_info = {line.rstrip().split('\t')[0]: [line.rstrip().split('\t')[1], line.rstrip().split('\t')[2]] for line in content}

	with open(args.training_fasta, 'r') as f:
		content = f.readlines()
		args.train_genomes_info = {line.rstrip().split('\t')[0]: [line.rstrip().split('\t')[1], line.rstrip().split('\t')[2]] for line in content}

	# verify that the genomes investigated only have one chromosome
	testing_fasta, _, training_fasta, training_records = CheckGenomes(args, args.pos_label)

	for label in args.neg_label:
		_, _, _, _ = CheckGenomes(args, label)

	# get reads in training set fasta file
	train_readid_to_read, train_sequence_length, _ = LoadFnaFile(args.training_fna_file)
	# get reads in testing set fasta file
	test_readid_to_read, test_sequence_length, test_ordered_reads_id = LoadFnaFile(args.testing_fna_file)
	
	# create output directories
	args.output_dir = os.path.join(os.getcwd(), args.pos_label)
	if not os.path.isdir(args.output_dir):
		os.makedirs(args.output_dir)
	if not os.path.isdir(os.path.join(args.output_dir, 'blast')):
		os.makedirs(os.path.join(args.output_dir, 'blast'))
	if not os.path.isdir(os.path.join(args.output_dir, 'Genomes_GTF_missing')):
		os.makedirs(os.path.join(args.output_dir, 'Genomes_GTF_missing'))

	# get FN and TP sequences
	fn_sequences = set()
	tp_sequences = set()
	with open(args.testing_results, 'r') as f:
		for count, line in enumerate(f):
			prob = float(line.rstrip().split('\t')[2])
			if prob >= args.prob_threshold:
				if line.rstrip().split('\t')[0] == '1' and line.rstrip().split('\t')[1] == '0':
					fn_sequences.add(test_ordered_reads_id[count])
				if line.rstrip().split('\t')[0] == '1' and line.rstrip().split('\t')[1] == '1':
					tp_sequences.add(test_ordered_reads_id[count])

	# blast testing reads to training genome (get average coverage for fn and tp reads)
	RunBlast(args, os.path.join(args.output_dir, 'blast', 'test_reads_train_genome'), args.testing_fna_file, subject=[training_fasta], outfilename=f'{args.output_dir}/blast/test_reads_train_genome/all_test_pos_train_blastn.out')
	test_reads_id = list(fn_sequences) + list(tp_sequences)
	test_alignments_pos_train = GetReadsAlignments(test_reads_id, f'{args.output_dir}/blast/test_reads_train_genome/all_test_pos_train_blastn.out', test_sequence_length, seq_to_labels, os.path.join(args.output_dir, f'blast/test_reads_train_genome/all_test_pos_train_{args.prob_threshold}_mapping_info.tsv'))
	
	# get training coverage
	train_coverage, _, _, _ = GetTrainCoverage(args, training_fasta, fn_sequences, tp_sequences, test_alignments_pos_train)

	# get info about genomic islands
	if os.path.isdir(args.genomic_islands):
		gis_align = GetGIsFromFasta(args, args.train_genomes_info[args.label_pos][0], training_fasta)
	else:
		gis_align = GetGIsFromAnnotations(args, input_dir, str(training_records[0].seq), args.train_genomes_info[args.label_pos][0], training_fasta)

	print(gis_align)
	list_testing_fasta = [args.test_genomes_info[l][1] for l in args.neg_label] + [testing_fasta]
	print(list_testing_fasta)
	CircosPlot(args, list_testing_fasta, training_fasta, train_coverage, \
		os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_coverage_circos.png'), genomic_islands=gis_align)



