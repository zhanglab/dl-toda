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
# from pygenomeviz.align import AlignCoord, Blast
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


def CheckGenomes(args):
	# load testing fasta file
	with open(args.testing_fasta, "r") as handle:
		test_records = list(SeqIO.parse(handle, "fasta"))

	# load training fasta file
	with open(args.train_genomes_info[args.label][1], "r") as handle:
		train_records = list(SeqIO.parse(handle, "fasta"))

	assert len(test_records) == 1, f'{arg.label}\t{args.testing_genome} has more than 1 chromosome'
	assert len(train_records) == 1, f'{arg.label}\t{args.train_genomes_info[args.label][0]} has more than 1 chromosome'

	genome_size = 0


	return args.testing_fasta, test_records, args.train_genomes_info[args.label][1], train_records


def GetGCSkew(sequence):
	all_gc_skew = []
	window_size = int(len(sequence) / 500)
	step_size = int(len(sequence) / 1000)
	pos_list = list(range(0, len(sequence), step_size)) + [len(sequence)]
	for pos in pos_list:
		start = pos - int(window_size / 2)
		end = pos + int(window_size / 2)
		
		# update start and end of window according to the size of the sequence
		if start < 0:
			start = 0

		if end > len(sequence):
			end = len(sequence)

		gcskew_seq = sequence[start:end]
		g_count = gcskew_seq.upper().count("G")
		c_count = gcskew_seq.upper().count("C")

		if float(g_count + c_count) == 0.0:
			gcskew = 0.0
		else:
			gcskew = (g_count - c_count) / float(g_count + c_count)

		all_gc_skew.append(gcskew)

	return np.array(pos_list).astype(np.int64), np.array(all_gc_skew).astype(np.float64)


def GetGCContent(sequence):
	all_gc_content = []
	window_size = int(len(sequence) / 500)
	step_size = int(len(sequence) / 1000)

	pos_list = list(range(0, len(sequence), step_size)) + [len(sequence)]
	for pos in pos_list:
		start = pos - int(window_size / 2)
		end = pos + int(window_size / 2)

		# update start and end of window according to the size of the sequence
		if start < 0:
			start = 0

		if end > len(sequence):
			end = len(sequence)

		gccontent_seq = sequence[start:end]
		gc_content = SeqUtils.gc_fraction(gccontent_seq) * 100
		all_gc_content.append(gc_content)

	genome_gc_content = SeqUtils.gc_fraction(sequence) * 100

	return np.array(pos_list).astype(np.int64), np.array(all_gc_content).astype(np.float64), genome_gc_content


def GetReadsGCcontent(args, gc_content, pos_list, alignments, type):
	reads_gc_content = []
	for readid, data in alignments.items():
		read_gc = []
		read_pos = []
		for i in range(0, len(pos_list)-1, 1):
			if (data[2] <= pos_list[i] and data[3] >= pos_list[i]) or \
				(data[2] >= pos_list[i] and data[3] <= pos_list[i+1]) or \
				(data[2] <= pos_list[i+1] and data[3] >= pos_list[i+1]) or \
				(data[2] <= pos_list[i] and data[3] >= pos_list[i+1]):
					read_gc.append(gc_content[i])
					read_pos.append([pos_list[i],pos_list[i+1]])
		ave_read_gc = sum(read_gc)/len(read_gc)
		if len(read_pos) > 1:
			print(f'{readid}\t{data}\t{read_gc}\t{ave_read_gc}\t{read_pos}')
		reads_gc_content.append(ave_read_gc)

	with open(os.path.join(args.output_dir, f'{args.label}_{type}_gc_content.tsv'), 'w') as f:
		f.write(f'#reads\t{len(reads_gc_content)}\n'
				f'mean\t{statistics.mean(reads_gc_content)}\n'
				f'median\t{statistics.median(reads_gc_content)}\n'
				f'min\t{min(reads_gc_content)}\n'
				f'max\t{max(reads_gc_content)}\n')
		

# def GetTrainCoverage(args, training_fasta, fn_sequences, tp_sequences, test_alignments_pos_train):
# 	# get reads in training set fasta file
# 	readid_to_read, readsid_to_length, _ = LoadFnaFile(args.training_fna_file)

# 	train_reads_id = []
# 	with open(os.path.join(args.output_dir, 'train_coverage', f'{args.label}_train_pos_reads.fq'), 'w') as outf:
# 		for k, v in readid_to_read.items():
# 			if k.split('|')[1] == args.label:
# 				outf.write(f'@{k}\n{v}\n+\n{len(v)*"J"}\n')
# 				train_reads_id.append(k)

# 	RunBowtie(args, training_fasta, os.path.join(args.output_dir, 'train_coverage', f'{args.label}_train_pos_reads.fq'), os.path.join(args.output_dir, 'train_coverage', f'{args.label}_pos_train_coverage.sam'))
	
# 	ref_info, alignments = LoadData(os.path.join(args.output_dir, 'train_coverage', f'{args.label}_pos_train_coverage.sam'))
# 	print(ref_info)

# 	ref = ref_info[0][0]
# 	length_ref = ref_info[0][1]
# 	dict_coverage, reads_info = GetCoverageOfSample(alignments[ref], length_ref, label=None)

# 	# get coverage per base
# 	base_coverage = [dict_coverage[i] for i in range(length_ref)]
# 	total_bases = sum(base_coverage)
# 	coverage = round(total_bases / length_ref, 3)

# 	with open(os.path.join(args.output_dir, 'train_coverage', f'{args.label}_{ref}_train_coverage.tsv'), 'w') as f:
# 		f.write(f'{total_bases}\t{length_ref}\t{coverage}')

# 	# get train coverage in position mapped by TP reads
# 	tp_cov = []
# 	fn_cov = []
# 	for read_id, data in test_alignments_pos_train.items():
# 		base_read_cov = [base_coverage[pos-1] for pos in range(data[2], data[3]+1, 1)]
# 		ave_read_cov = round(sum(base_read_cov)/(data[3]-data[2]), 3)
# 		if read_id in tp_sequences:
# 			tp_cov.append(ave_read_cov)
# 		elif read_id in fn_sequences:
# 			fn_cov.append(ave_read_cov)

# 	with open(os.path.join(args.output_dir, f'{args.label}_test_train_average_coverage.tsv'), 'w') as f:
# 		f.write(f'TP\t{len(tp_cov)}\t{len(tp_sequences)}\t{round(len(tp_cov)/len(tp_sequences),3)*100}\t{statistics.mean(tp_cov)}\t{statistics.median(tp_cov)}\t{min(tp_cov)}\t{max(tp_cov)}\n')
# 		f.write(f'FN\t{len(fn_cov)}\t{len(fn_sequences)}\t{round(len(fn_cov)/len(fn_sequences),3)*100}\t{statistics.mean(fn_cov)}\t{statistics.median(fn_cov)}\t{min(fn_cov)}\t{max(fn_cov)}\n')

# 	return base_coverage, ref_info, train_reads_id, readsid_to_length
	

def LoadFnaFile(fasta_file):
	with open(fasta_file, 'r') as f:
		content = f.readlines()
	ordered_reads_id = [content[i].rstrip()[1:] for i in range(0,len(content),2)]
	readsid_to_seq = dict(zip([content[i].rstrip()[1:] for i in range(0, len(content), 2)], [content[i].rstrip() for i in range(1, len(content), 2)]))
	readsid_to_length = dict(zip([content[i].rstrip()[1:] for i in range(0, len(content), 2)], [len(content[i].rstrip()) for i in range(1, len(content), 2)]))

	return readsid_to_seq, readsid_to_length, ordered_reads_id


def ConcatenateFiles(list_files, outfilename, data_type):
	if data_type == 'function':
		concatenated_data = defaultdict(int)
		for input_file in list_files:
			with open(input_file, 'r') as inf:
				for line in inf:
					concatenated_data[line.rstrip().split('\t')[0]] += int(line.rstrip().split('\t')[1])
		
		concatenated_data_sorted = dict(sorted(concatenated_data.items(), key=lambda item: item[1], reverse=True))
		with open(outfilename, 'w') as outf:
			for k, v in concatenated_data_sorted.items():
				outf.write(f'{k}\t{v}\n')

	elif data_type == 'gene_type':
		concatenated_data = defaultdict(int)
		for input_file in list_files:
			with open(input_file, 'r') as inf:
				for line in inf:
					concatenated_data['protein_coding'] += int(line.rstrip().split('\t')[0])
					concatenated_data['tRNA'] += int(line.rstrip().split('\t')[1])
					concatenated_data['rRNA'] += int(line.rstrip().split('\t')[2])

		with open(outfilename, 'w') as outf:
			outf.write(f'{concatenated_data["protein_coding"]}\t{concatenated_data["tRNA"]}\t{concatenated_data["rRNA"]}\n')

	else:
		with open(outfilename, 'w') as outf:
			for input_file in list_files:
				with open(input_file, 'r') as inf:
					outf.write(inf.read())


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


def GetFNOtherInfo(args, pos_test_alignments, neg_train_alignments, annot_info, sequence_length, readid_to_read):
	""" get genes on testing genome associated wth FN reads and taxa that were mapped by FN reads """
	taxa = defaultdict(int)
	list_reads_id = [] # store id of FN reads mapped to training genomes from the label 0
	reads_wo_genes = []
	with open(os.path.join(args.output_dir, f'fn_pos_test_neg_train_{args.prob_threshold}_summary.tsv'), 'w') as f:
		for readid, data in neg_train_alignments.items():
			taxa[data[0]] += 1
			list_reads_id.append(readid)
			if readid in pos_test_alignments:
				genes = defaultdict(str)
				for pos in range(pos_test_alignments[readid][2], pos_test_alignments[readid][3]+1, 1):
					for gene_id, annot in annot_info.items():
						if pos >= annot[1] and pos <= annot[2]:
							genes[gene_id] = annot[4]
				f.write(f"{readid}\t{sequence_length[readid]}\t{pos_test_alignments[readid][2]}\t{pos_test_alignments[readid][3]+1}\t{data[1]}\t{data[0]}\t{args.dl_toda_tax[data[0]]}")
				if len(genes) != 0:
					for gene_id in genes.keys():
						f.write(f'\t{gene_id}\t{genes[gene_id]}')
				else:
					reads_wo_genes.append(readid)
					f.write('\tNA')
				f.write('\n')

	taxa_sorted = dict(sorted(taxa.items(), key=lambda item: item[1], reverse=True))
	with open(os.path.join(args.output_dir, f'fn_pos_test_neg_train_{args.prob_threshold}_mapped_taxa.tsv'), 'w') as f:
		for count, (k, v) in enumerate(taxa_sorted.items()):
			f.write(f'{k}\t{args.dl_toda_tax[k]}\t{v}\n')


	return list_reads_id


def CreateTsvFile(reads_id, readid_to_read, filename):	
	with open(filename, 'w') as f:
		f.write(''.join([f'>{r}\n{readid_to_read[r]}\n' for r in list(reads_id)]))


def GetAnnotInfo(args, genome_id, input_dir):
	if f'{genome_id}_gtf' not in os.listdir(args.annotations_dir):
		annot_output_dir = os.path.join(args.annotations_dir, f'{genome_id}_gtf')
		os.makedirs(annot_output_dir)
		os.chdir(annot_output_dir)
		# download feature table in gtf if not present
		result = subprocess.run([ncbi_datasets_exec, 'download', 'genome', 'accession', f'{genome_id}', '--include', 'gtf'])
		# unzip output folder
		with zipfile.ZipFile('ncbi_dataset.zip', 'r') as zip_ref:
			zip_ref.extractall(os.getcwd())
		os.chdir(input_dir)
	else:
		print(f'{genome_id}\tdownload already done')

	
	annot_file = glob.glob(os.path.join(args.annotations_dir, f'{genome_id}_gtf/ncbi_dataset/data/{genome_id}/genomic.gtf'))
	if len(annot_file) == 0:
		f = open(os.path.join(args.output_dir, 'Genomes_GTF_missing', f'{genome_id}.txt'), 'w')
		f.close()
		return {}, {}
	else:
		genes_type = defaultdict(str)
		annot_info = defaultdict(list)
		locus_tags_info = defaultdict(list)
		with open(annot_file[0], 'r') as f:
			content = f.readlines()
			for i in range(5,len(content)-1,1):
				begin = int(content[i].rstrip().split('\t')[3])
				end = int(content[i].rstrip().split('\t')[4])
				strand = content[i].rstrip().split('\t')[6]
				gene_id = ''
				gene = ''
				biotype = ''
				function = ''
				old_locus_tag = ''
				for e in content[i].rstrip().split('\t')[8].split(';'):
					e = e.replace('"', '')
					# get all go_function entries and choose go_function with the most details
					if 'go_function' in e:
						fn = e.split('|')[0].split(' ')[2:]
						if len(fn) > len(function):
							function = ' '.join(fn)
					if 'product' in e:
						gene = ' '.join(e.split(' ')[2:])
					if 'gene_id' in e:
						gene_id = e.split(' ')[1]
					if 'gene_biotype' in e:
						biotype = e.split(' ')[2]
					if 'old_locus_tag' in e:
						old_locus_tag = e.split(' ')[2]

				if content[i].rstrip().split('\t')[2] == 'gene':
					genes_type[gene_id] = biotype
					locus_tags_info[gene_id] = [begin, end, old_locus_tag, strand]
				elif content[i].rstrip().split('\t')[2] == 'CDS' and genes_type[gene_id] == 'protein_coding':
					if function == '':
						function = gene
					annot_info[gene_id] = ['protein_coding', begin, end, strand, gene, function]
				elif content[i].rstrip().split('\t')[2] == 'transcript' and genes_type[gene_id] == 'tRNA':
					annot_info[gene_id] = ['tRNA', begin, end, strand, gene]
				elif content[i].rstrip().split('\t')[2] == 'transcript' and genes_type[gene_id] == 'rRNA':
					annot_info[gene_id] = ['rRNA', begin, end, strand, gene]
				
				assert gene_id != '', 'gene id should not be unknown'
		
		print(len([k for k, v in annot_info.items() if v[0] == 'protein_coding']))
		print(len([k for k, v in annot_info.items() if v[0] == 'tRNA']))
		print(len([k for k, v in annot_info.items() if v[0] == 'rRNA']))

		return annot_info, locus_tags_info



def GetReadsForAttentions(args, tp_alignments_pos_test, fn_alignments_pos_test, test_readid_to_read):
	reads = []
	reads_id = {}
	for fn_readid, fn_data in fn_alignments_pos_test.items():
		if fn_data[2] < fn_data[3]:
			fn_start_pos = fn_data[2]
			fn_end_pos = fn_data[3]
		else:
			fn_start_pos = fn_data[3]
			fn_end_pos = fn_data[2]
		fn_strand = fn_data[6]

		for tp_readid, tp_data in tp_alignments_pos_test.items():
			if tp_data[2] < tp_data[3]:
				tp_start_pos = tp_data[2]
				tp_end_pos = tp_data[3]
			else:
				tp_start_pos = tp_data[3]
				tp_end_pos = tp_data[2]
			tp_strand = tp_data[6]

			if (tp_start_pos < fn_end_pos and tp_end_pos > fn_start_pos) or \
				(fn_start_pos < tp_end_pos and fn_end_pos > tp_start_pos) or \
				(tp_start_pos < fn_start_pos and tp_end_pos > fn_end_pos) or \
				(fn_start_pos < tp_start_pos and fn_end_pos > tp_end_pos):
				if tp_strand == 'plus' and fn_strand == 'plus':
					if abs(len(test_readid_to_read[fn_readid])-len(test_readid_to_read[tp_readid])) < 200:
						reads.append([tp_readid.split('|')[2], f'{tp_readid}-tp-{tp_start_pos}-{tp_end_pos}', len(test_readid_to_read[tp_readid]), tp_strand, \
							fn_readid.split('|')[2], f'{fn_readid}-fn-{fn_start_pos}-{fn_end_pos}', len(test_readid_to_read[fn_readid]), fn_strand])
						reads_id[tp_readid] = f'{tp_readid}-tp-{tp_start_pos}-{tp_end_pos}'
						reads_id[fn_readid] = f'{fn_readid}-fn-{fn_start_pos}-{fn_end_pos}'

	tsv_file = open(os.path.join(args.output_dir, f'{args.label}_contiguous_fn_tp_reads.tsv'), 'w')
	sum_file = open(os.path.join(args.output_dir, f'{args.label}_contiguous_fn_tp_id.tsv'), 'w')
	for r in reads:
		sum_file.write(f'{r[0]}')
		for idx in range(1, len(r), 1):
			sum_file.write(f'\t{r[idx]}')
		sum_file.write('\n')
	for k, v in reads_id.items():
		tsv_file.write(f'{v}\t{test_readid_to_read[k]}\n')
	tsv_file.close()
	sum_file.close()


def CheckReadInGene(read_start_pos, read_end_pos, gene_start_pos, gene_end_pos):
	# check if read_id maps to gene
	length_mapped_seq = 0
	if (read_start_pos <= gene_start_pos and read_end_pos >= gene_end_pos) or \
		(read_start_pos <= gene_start_pos and read_end_pos >= gene_start_pos) or \
		(read_start_pos >= gene_start_pos and read_end_pos <= gene_end_pos) or \
		(read_start_pos <= gene_end_pos and read_end_pos >= gene_end_pos):
		if (read_start_pos <= gene_start_pos and read_end_pos >= gene_end_pos):
			length_mapped_seq = 100
		elif (read_start_pos <= gene_start_pos and read_end_pos >= gene_start_pos):
			length_mapped_seq = (read_end_pos - gene_start_pos)/(gene_end_pos - gene_start_pos)*100
		elif (read_start_pos >= gene_start_pos and read_end_pos <= gene_end_pos):
			length_mapped_seq = (read_end_pos - read_start_pos)/(gene_end_pos - gene_start_pos)*100
		elif (read_start_pos <= gene_end_pos and read_end_pos >= gene_end_pos):
			length_mapped_seq = (gene_end_pos - read_start_pos)/(gene_end_pos - gene_start_pos)*100
	return length_mapped_seq


def GetGenes(args, annot_info, fn_alignments, tp_alignments, sequence_length, readid_to_read, genome_size, fn_cs, tp_cs):
	# get length and function of fn sequences per mapped position on the genome investigated
	tp_genes = defaultdict(list)
	fn_genes = defaultdict(list)
	fn_functions = defaultdict(int)
	tp_functions = defaultdict(int)
	fn_reads_kept = [] 
	tp_reads_kept = []
	sel_tp_evalue = dict()
	sel_tp_pident = dict()
	sel_fn_evalue = dict()
	sel_fn_pident = dict()
	scores = {i:0 for i in range(genome_size)}

	for gene_id, data in annot_info.items():
		gene_start_pos = data[1]
		gene_end_pos = data[2]
		fn_reads = []
		tp_reads = []
		tp_evalue = dict()
		tp_pident = dict()
		fn_evalue = dict()
		fn_pident = dict()
		for read_id, align_info in fn_alignments.items():
			if align_info[2] < align_info[3]:
				read_start_pos = align_info[2]
				read_end_pos = align_info[3]
			else:
				read_start_pos = align_info[3]
				read_end_pos = align_info[2]
			length_mapped_seq = CheckReadInGene(read_start_pos, read_end_pos, gene_start_pos, gene_end_pos)
			if length_mapped_seq > 0:
				fn_reads.append(read_id)
				fn_evalue[read_id] = align_info[4]
				fn_pident[read_id] = align_info[5]

		for read_id, align_info in tp_alignments.items():
			if align_info[2] < align_info[3]:
				read_start_pos = align_info[2]
				read_end_pos = align_info[3]
			else:
				read_start_pos = align_info[3]
				read_end_pos = align_info[2]
			length_mapped_seq = CheckReadInGene(read_start_pos, read_end_pos, gene_start_pos, gene_end_pos)
			if length_mapped_seq > 0:
				tp_reads.append(read_id)
				tp_evalue[read_id] = align_info[4]
				tp_pident[read_id] = align_info[5]

		if len(fn_reads) + len(tp_reads) > 0:
			# compare number of tp and fn positions mapped to gene
			fn_num_pos = sum([sequence_length[r] for r in fn_reads])
			tp_num_pos = sum([sequence_length[r] for r in tp_reads])


			ratio_fn = round(fn_num_pos / (tp_num_pos + fn_num_pos), 2)
			ratio_tp = round(tp_num_pos / (tp_num_pos + fn_num_pos), 2)
			if ratio_fn > 0.5:
				fn_genes[gene_id] = [ratio_fn, fn_num_pos, tp_num_pos, len(fn_reads), len(tp_reads), gene_start_pos, gene_end_pos]
				for i in range(gene_start_pos, gene_end_pos+1, 1):
					scores[i-1] = ratio_fn
				fn_reads_kept += fn_reads
				sel_fn_evalue.update(fn_evalue)
				sel_fn_pident.update(fn_pident)
				if data[0] == 'protein_coding':
					fn_functions[data[5]] += 1

			elif ratio_tp > 0.5:
				tp_genes[gene_id] = [ratio_tp, fn_num_pos, tp_num_pos, len(fn_reads), len(tp_reads), gene_start_pos, gene_end_pos]
				tp_reads_kept += list(tp_reads)
				sel_tp_evalue.update(tp_evalue)
				sel_tp_pident.update(tp_pident)
				if data[0] == 'protein_coding':
					tp_functions[data[5]] += 1

	fn_functions_sorted = dict(sorted(fn_functions.items(), key=lambda item: item[1], reverse=True))
	with open(os.path.join(args.output_dir, f'{args.label}_fn_functions_{args.prob_threshold}.tsv'), 'w') as f:
		for k, v in fn_functions_sorted.items():
			f.write(f'{k}\t{v}\n')

	tp_functions_sorted = dict(sorted(tp_functions.items(), key=lambda item: item[1], reverse=True))
	with open(os.path.join(args.output_dir, f'{args.label}_tp_functions_{args.prob_threshold}.tsv'), 'w') as f:
		for k, v in tp_functions_sorted.items():
			f.write(f'{k}\t{v}\n')

	sel_fn_cs = [str(fn_cs[r]) for r in fn_reads_kept]
	with open(os.path.join(args.output_dir, f'{args.label}_selected_fn_cs_{args.prob_threshold}.tsv'), 'w') as f:
		f.write('\n'.join(sel_fn_cs))
	
	sel_tp_cs = [str(tp_cs[r]) for r in tp_reads_kept]
	with open(os.path.join(args.output_dir, f'{args.label}_selected_tp_cs_{args.prob_threshold}.tsv'), 'w') as f:
		f.write('\n'.join(sel_tp_cs))

	sel_fn_length = [str(sequence_length[r]) for r in fn_reads_kept]
	with open(os.path.join(args.output_dir, f'{args.label}_selected_fn_length_{args.prob_threshold}.tsv'), 'w') as f:
		f.write('\n'.join(sel_fn_length))

	sel_tp_length = [str(sequence_length[r]) for r in tp_reads_kept]
	with open(os.path.join(args.output_dir, f'{args.label}_selected_tp_length_{args.prob_threshold}.tsv'), 'w') as f:
		f.write('\n'.join(sel_tp_length))


	with open(os.path.join(args.output_dir, f'{args.label}_tp_genes_{args.prob_threshold}.tsv'), 'w') as f:
		for k, v in tp_genes.items():
			f.write(f'{k}')
			for i in range(len(annot_info[k])):
				f.write(f'\t{annot_info[k][i]}')
			for i in range(len(v)):
				f.write(f'\t{v[i]}')
			f.write('\n')

	with open(os.path.join(args.output_dir, f'{args.label}_fn_genes_{args.prob_threshold}.tsv'), 'w') as f:
		for k, v in fn_genes.items():
			f.write(f'{k}')
			for i in range(len(annot_info[k])):
				f.write(f'\t{annot_info[k][i]}')
			for i in range(len(v)):
				f.write(f'\t{v[i]}')
			f.write('\n')

	scores_list = [scores[i] for i in range(genome_size)]

	with open(os.path.join(args.output_dir, f'{args.label}_fn_tp_scores_info.tsv'), 'w') as outf:
		outf.write(f'# fn reads kept: {len(fn_reads_kept)}\n')
		outf.write(f'# tp reads kept: {len(tp_reads_kept)}\n')
		outf.write(f'FN rate all positions:\tmean:{statistics.mean(scores_list)}\tmedian:{statistics.median(scores_list)}\tmin:{min(scores_list)}\tmax:{max(scores_list)}\n')
		outf.write(f'fn evalue:\tmean:{statistics.mean(sel_fn_evalue.values())}\tmedian:{statistics.median(sel_fn_evalue.values())}\tmin:{min(sel_fn_evalue.values())}\tmax:{max(sel_fn_evalue.values())}\n')
		outf.write(f'tp evalue:\tmean:{statistics.mean(sel_tp_evalue.values())}\tmedian:{statistics.median(sel_tp_evalue.values())}\tmin:{min(sel_tp_evalue.values())}\tmax:{max(sel_tp_evalue.values())}\n')
		outf.write(f'fn pident:\tmean:{statistics.mean(sel_fn_pident.values())}\tmedian:{statistics.median(sel_fn_pident.values())}\tmin:{min(sel_fn_pident.values())}\tmax:{max(sel_fn_pident.values())}\n')
		outf.write(f'tp pident:\tmean:{statistics.mean(sel_tp_pident.values())}\tmedian:{statistics.median(sel_tp_pident.values())}\tmin:{min(sel_tp_pident.values())}\tmax:{max(sel_tp_pident.values())}\n')

	# create tsv files with FN and TP reads
	CreateTsvFile(fn_reads_kept, readid_to_read, os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_fn_reads_genes.tsv'))
	CreateTsvFile(tp_reads_kept, readid_to_read, os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_tp_reads_genes.tsv'))

	return scores_list, fn_genes, tp_genes



# def GetGenes(args, label, output_dir, annot_info, fn_alignments, fn_reads_kept, sequence_length, readid_to_read, type):
# 	# get length and function of fn sequences per mapped position on the genome investigated
# 	genes = defaultdict(list)
# 	functions = defaultdict(int)
# 	genestype = defaultdict(int)
# 	readid_w_gene = defaultdict(list)
# 	# pos_readid = defaultdict(list) # key: position in target genome, value: list of reads id mapped to that position

# 	for readid in fn_reads_kept:
# 		data = fn_alignments[readid]
# 		if data[2] < data[3]:
# 			start_pos = data[2]
# 			end_pos = data[3]
# 		else:
# 			start_pos = data[3]
# 			end_pos = data[2]
# 		# for pos in range(start_pos, end_pos+1, 1):
# 		# 	pos_readid[pos-1].append(readid)
# 		for gene_id, annot in annot_info.items():
# 			if (start_pos <= annot[1] and end_pos >= annot[2]) or \
# 			(start_pos <= annot[1] and end_pos >= annot[1]) or \
# 			(start_pos >= annot[1] and end_pos <= annot[2]) or \
# 			(start_pos <= annot[2] and end_pos >= annot[2]):
# 				if (start_pos <= annot[1] and end_pos >= annot[2]):
# 					length_mapped_seq = 100
# 				elif (start_pos <= annot[1] and end_pos >= annot[1]):
# 					length_mapped_seq = (end_pos - annot[1])/(annot[2]- annot[1])*100
# 				elif (start_pos >= annot[1] and end_pos <= annot[2]):
# 					length_mapped_seq = (end_pos - start_pos)/(annot[2]- annot[1])*100
# 				elif (start_pos <= annot[2] and end_pos >= annot[2]):
# 					length_mapped_seq = (annot[2] - start_pos)/(annot[2]- annot[1])*100
# 				if annot[0] == 'protein_coding':
# 					functions[annot[5]] += 1
# 				genes[gene_id] = annot
# 				readid_w_gene[readid] = [data[2], data[3], gene_id, length_mapped_seq]
# 				genestype[annot[0]] += 1

# 	# pos_readid_count = [len(v) for v in pos_readid.values()]
	
# 	# if len(pos_readid_count) > 0:
# 	# 	print(f'mean: {statistics.mean(pos_readid_count)}\tmedian: {statistics.median(pos_readid_count)}\tmin: {min(pos_readid_count)}\tmax: {max(pos_readid_count)}')

# 	with open(os.path.join(output_dir, f'{label}_fn_reads_kept_alignments_{args.prob_threshold}.tsv'), 'w') as outf:
# 		for readid, data in readid_w_gene.items():
# 			outf.write(f'{readid}\t{data[0]}\t{data[1]}\t{data[2]}\n')

# 	genes_of_interest = defaultdict(list)
# 	genes_of_interest_count = defaultdict(int)
# 	genes_of_interest_stat = defaultdict(list)
# 	for readid in readid_w_gene.keys():
# 		gene_id = readid_w_gene[readid][2]
# 		genes_of_interest[gene_id] = genes[gene_id]
# 		genes_of_interest_count[gene_id] += 1
# 		genes_of_interest_stat[gene_id] += [readid_w_gene[readid][3]]

# 	# for pos, list_readid in pos_readid.items():
# 	# 	if len(list_readid) >= 3:
# 	# 		for readid in list_readid:
# 	# 			if readid in readid_w_gene:
# 	# 				gene_id = readid_w_gene[readid][2]
# 	# 				genes_of_interest[gene_id] = genes[gene_id]

# 	reads_wo_genes = []
# 	if len(readid_w_gene) != len(fn_alignments):
# 		with open(os.path.join(output_dir, f'{label}_{type}_reads_wo_gene_{args.prob_threshold}.tsv'), 'w') as f:
# 			for readid, data in fn_alignments.items():
# 				if readid not in readid_w_gene:
# 					f.write(f'{readid}\t{sequence_length[readid]}\t{data[0]}\t{data[1]}\t{data[2]}\t{data[3]}\n')
# 					reads_wo_genes.append(readid)

# 		with open(os.path.join(output_dir, f'{label}_{type}_wo_gene_{args.prob_threshold}.fna'), 'w') as f:
# 			f.write(''.join([f'>{r}\n{readid_to_read[r]}\n' for r in reads_wo_genes]))
# 	else:
# 		print('all reads were found a gene')

# 	# with open(os.path.join(output_dir, f'{label}_{type}_genes_info_{args.prob_threshold}.tsv'), 'w') as outf:
# 	# 	for gene_id, annot in genes_of_interest.items():
# 	# 		if annot[0] == 'protein_coding':
# 	# 			outf.write(f'{gene_id}\t{annot[3]}\t{annot[1]}\t{annot[2]}\t{annot[4]}\t{annot[0]}\t{annot[5]}\t{genes_of_interest_count[gene_id]}\t{statistics.mean(genes_of_interest_stat[gene_id])}\n')
# 	# 		else:
# 	# 			outf.write(f'{gene_id}\t{annot[3]}\t{annot[1]}\t{annot[2]}\t{annot[4]}\t{annot[0]}\t{genes_of_interest_count[gene_id]}\t{statistics.mean(genes_of_interest_stat[gene_id])}\n')

# 	functions_sorted = dict(sorted(functions.items(), key=lambda item: item[1], reverse=True))
# 	with open(os.path.join(output_dir, f'{label}_{type}_functions_{args.prob_threshold}.tsv'), 'w') as f:
# 		for k, v in functions_sorted.items():
# 			f.write(f'{k}\t{v}\n')

# 	genestype_sorted = dict(sorted(genestype.items(), key=lambda item: item[1], reverse=True))
# 	with open(os.path.join(output_dir, f'{label}_{type}_genes_type_{args.prob_threshold}.tsv'), 'w') as f:
# 		f.write(f'{genestype["protein_coding"]}\t{genestype["tRNA"]}\t{genestype["rRNA"]}\n')

# 	return genes_of_interest, genes_of_interest_count, genes_of_interest_stat


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
				strand = line.rstrip().split(',')[11]
				if readid in alignments:
					if evalue < alignments[readid][4] and pident > alignments[readid][5]:
						alignments[readid] = [seq_label, seq_id, sstart, send, evalue, pident, strand]						
				else:
					alignments[readid] = [seq_label, seq_id, sstart, send, evalue, pident, strand]

	if outfilename:
		with open(outfilename, 'w') as f:
			unmapped_reads_id = list(sequences.difference(set(alignments.keys())))
			if len(unmapped_reads_id) != 0:
				unmapped_reads_length = [sequence_length[r] for r in unmapped_reads_id]
				f.write(f'# unmapped reads:\t{len(unmapped_reads_id)}\nmean reads length:\t{statistics.mean(unmapped_reads_length)}\nmedian reads length:\t{statistics.median(unmapped_reads_length)}\nmin reads length:\t{min(unmapped_reads_length)}\nmax reads length:\t{max(unmapped_reads_length)}')
			else:
				f.write(f'# unmapped reads:\t{len(unmapped_reads_id)}\nmean reads length:\tNA\nmedian reads length:\tNA\nmin reads length:\tNA\nmax reads length:\tNA')

	return alignments


# def GetScores(args, testing_records, tp_alignments, fn_alignments):
# 	genome_size = len(testing_records[0].seq)
# 	# shannon_scores = []
# 	scores = []
# 	fn_scores = []
# 	fn_reads_kept = []
# 	tp_reads_kept = []
# 	sel_tp_evalue = dict()
# 	sel_tp_pident = dict()
# 	sel_fn_evalue = dict()
# 	sel_fn_pident = dict()
# 	for i in range(1, genome_size+1, 1):
# 		num_tp = 0
# 		num_fn = 0

# 		fn_reads = set()
# 		tp_reads = set()
# 		tp_evalue = dict()
# 		tp_pident = dict()
# 		fn_evalue = dict()
# 		fn_pident = dict()
		
# 		# check if position is located in a read assigned to TP
# 		for read_id, data in tp_alignments.items():
# 			if data[4] == 0 and data[5] == 100 :
# 				if i >= data[2] and i <= data[3]:
# 					tp_reads.add(read_id)
# 					tp_evalue[read_id] = data[4]
# 					tp_pident[read_id] = data[5]
# 					num_tp += 1

# 		# check if position is located in a read assigned to FN
# 		for read_id, data in fn_alignments.items():
# 			if data[4] == 0 and data[5] == 100 :
# 				if i >= data[2] and i <= data[3]:
# 					fn_reads.add(read_id)
# 					fn_evalue[read_id] = data[4]
# 					fn_pident[read_id] = data[5]
# 					num_fn += 1

# 		if num_tp+num_fn > 0:
# 			ratio_fn = num_fn / (num_tp+num_fn)
# 			ratio_tp = num_tp / (num_tp+num_fn)

# 			if ratio_fn > 0.5:
# 				scores.append(ratio_fn)
# 				fn_scores.append(ratio_fn)
# 				fn_reads_kept += list(fn_reads)
# 				sel_fn_evalue.update(fn_evalue)
# 				sel_fn_pident.update(fn_pident)
# 			elif ratio_tp > 0.5:
# 				tp_reads_kept += list(tp_reads)
# 				sel_tp_evalue.update(tp_evalue)
# 				sel_tp_pident.update(tp_pident)
# 				scores.append(0)
# 		else:
# 			# no fn or tp with that position
# 			scores.append(0)

# 		# # compute probability for each group
# 		# if num_tp+num_fn > 0:
# 		# 	prob_tp = num_tp / (num_tp+num_fn)
# 		# 	prob_fn = num_fn / (num_tp+num_fn)

# 		# 	# compute tp and fn contribution to shannon score
# 		# 	shannon_tp = prob_tp*math.log(prob_tp, 2) if prob_tp > 0 else 0
# 		# 	shannon_fn = prob_tp*math.log(prob_fn, 2) if prob_fn > 0 else 0

# 		# 	# compute shannon entropy
# 		# 	if (shannon_tp + shannon_fn) == 0:
# 		# 		# cases where the position exists only in TP or FN reads
# 		# 		shannon_entropy = 0
# 		# 	else:
# 		# 		# cases where the position exists in TP and FN reads
# 		# 		shannon_entropy = -(shannon_tp + shannon_fn)
# 		# 		assert shannon_entropy < 1, f'{num_tp}\t{prob_tp}\t{shannon_tp}\t{num_fn}\t{prob_fn}\t{shannon_fn}\t{shannon_entropy}'

# 		# else:
# 		# 	shannon_entropy = 0

# 		# scores.append(shannon_entropy)


# 	return scores, list(set(fn_reads_kept)), list(set(tp_reads_kept))


def GetMatchRegions(args, input_file, genomic_islands, identity_thr=MIN_IDENTITY):
	# ref_to_query = defaultdict(dict)
	# ref_start_end = defaultdict(list)
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


def StoreCS(args, dict_cs, type):
	list_cs = []
	for k, v in dict_cs.items():
		list_cs.append(v)
	with open(os.path.join(args.output_dir, f'{args.label}_{type}_{args.prob_threshold}.tsv'), 'w') as f:
		f.write('\n'.join([str(x) for x in list_cs]))


def GetSeqLength(args, sequences_id, sequence_length, type):
	if len(sequences_id) != 0:
		seq_length_info = [sequence_length[s] for s in sequences_id]
		print(f'{type}\tmean: {statistics.mean(seq_length_info)}\tmedian: {statistics.median(seq_length_info)}\tmax: {max(seq_length_info)}\tmin: {min(seq_length_info)}')

		with open(os.path.join(args.output_dir, f'{args.label}_{type}_{args.prob_threshold}_seq_length.tsv'), 'w') as f:
			f.write('\n'.join([str(x) for x in seq_length_info]))

			# f.write(f'{statistics.mean(seq_length_info)}\t{statistics.median(seq_length_info)}\t{max(seq_length_info)}\t{min(seq_length_info)}')

def GetGenomesInfo(fasta):
	with open(fasta, 'r') as f:
		content = f.readline()
	strain = ' '.join([e for e in content.split(',')[0].split(' ')[1:] if e not in ['chromosome', 'strain']])
	
	return strain


def CircosPlot(args, scores, test_record_seq, train_record_seq, testing_fasta, training_fasta, \
			fn_alignments_pos_test, tp_alignments_pos_test, outfigpath, genomic_islands=None):
	
	# load data from training and testing genomes of label 1
	query_fasta = Fasta(testing_fasta) # query --> testing genome
	ref_fasta = Fasta(training_fasta) # ref/subject --> training genome

	# Initialize circos instance
	circos = Circos(
	    sectors=query_fasta.get_seqid2size(),
	    # space=0 if len(ref_fasta.get_seqid2size()) == 1 else 2,
		space=10,
	)

	train_strain = GetGenomesInfo(training_fasta)
	test_strain = GetGenomesInfo(testing_fasta)
	circos.text(f'{test_strain}\n{query_fasta.full_genome_length:,} bp\n(testing genome)', size=9, r=22)

	# print(f"Ref: {ref_fasta.name}\n({ref_fasta.full_genome_length:,} bp)\n{ref_fasta.full_genome_length}")
	# print(f"Query: {query_fasta.name}\n({query_fasta.full_genome_length:,} bp)\n{query_fasta.full_genome_length}")
	with open(os.path.join(args.output_dir, f'{args.label}_genomes_length.tsv'), 'w') as f:
		f.write(f'Testing genome:\t{query_fasta.name}\t{query_fasta.full_genome_length}\n')
		f.write(f'Training genome:\t{ref_fasta.name}\t{ref_fasta.full_genome_length}\n')

	min_r_pos = 100
	for sector in circos.sectors:
		# Plot labels of genomic islands
		if genomic_islands:
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
		if genomic_islands:
			outer_track.xticks_by_interval(TICKS_INTERVAL, label_formatter=lambda v: f"{v/1000000:.1f} Mb", outer=False,)
			min_r_pos -= 6
		else:
			outer_track.xticks_by_interval(TICKS_INTERVAL, label_formatter=lambda v: f"{v/1000000:.1f} Mb",)
			min_r_pos -= 1
		outer_track.xticks_by_interval(100000, tick_length=1, show_label=False)

		

		# create tracks for genomics features
		# f_cds_track = sector.add_track((min_r_pos-5, min_r_pos))
		# f_cds_track.axis(fc="lightgrey", ec="none", alpha=0.5)
		# min_r_pos -= 5
		# r_cds_track = sector.add_track((min_r_pos-5, min_r_pos))
		# r_cds_track.axis(fc="lightgrey", ec="none", alpha=0.5)
		# min_r_pos -= 5
		# Plot forward/reverse strand CDS
		
		# cds_track = sector.add_track((min_r_pos-4, min_r_pos))
		# min_r_pos -= 4
		# # rrna_track = sector.add_track((min_r_pos-5, min_r_pos))
		# # min_r_pos -= 6
		# trna_track = sector.add_track((min_r_pos-5, min_r_pos))
		# min_r_pos -= 6
		# features = []
		# features = {}
		# for gene_id in genes_of_interest.keys():
		# 	if genes_of_interest[gene_id][3] == '+':
		# 		location = FeatureLocation(start=genes_of_interest[gene_id][1], end=genes_of_interest[gene_id][2], strand=+1)
		# 		if genes_of_interest[gene_id][0] == 'protein_coding':
		# 			feature = SeqFeature(location=location, qualifiers={"gene_type": [genes_of_interest[gene_id][0]], "gene_id": [gene_id], "gene_name": [genes_of_interest[gene_id][4]], "strand": ["plus"], "function": [genes_of_interest[gene_id][5]]})
		# 			# cds_track.genomic_features(feature, plotstyle="arrow", fc="red")
		# 		else:
		# 			feature = SeqFeature(location=location, qualifiers={"gene_type": [genes_of_interest[gene_id][0]], "gene_id": [gene_id], "gene_name": [genes_of_interest[gene_id][4]], "strand": ["plus"]})
		# 			# if genes_of_interest[gene_id][0] == 'tRNA':
		# 				# cds_track.genomic_features(feature, fc="darkgreen")
		# 			# if genes_of_interest[gene_id][0] == 'rRNA':
		# 			# 	rrna_track.genomic_features(feature, fc="deeppink")
		# 	else:
		# 		location = FeatureLocation(start=genes_of_interest[gene_id][1], end=genes_of_interest[gene_id][2], strand=-1)
		# 		if genes_of_interest[gene_id][0] == 'protein_coding':
		# 			feature = SeqFeature(location=location, qualifiers={"gene_type": [genes_of_interest[gene_id][0]], "gene_id": [gene_id], "gene_name": [genes_of_interest[gene_id][4]], "strand": ["minus"], "function": [genes_of_interest[gene_id][5]]})
		# 			# cds_track.genomic_features(feature, plotstyle="arrow", fc="blue")
		# 		else:
		# 			feature = SeqFeature(location=location, qualifiers={"gene_type": [genes_of_interest[gene_id][0]], "gene_id": [gene_id], "gene_name": [genes_of_interest[gene_id][4]], "strand": ["minus"]})
		# 			# if genes_of_interest[gene_id][0] == 'tRNA':
		# 				# cds_track.genomic_features(feature, fc="darkgreen")
		# 			# if genes_of_interest[gene_id][0] == 'rRNA':
		# 			# 	rrna_track.genomic_features(feature, fc="deeppink")

		# 	# features.append(feature)
		# 	features[genes_of_interest[gene_id][1]] = feature

		
		# # Plot labels of genomic islands
		# if genomic_islands:
		# 	color = 'red'
		# 	# add track for genomic islands
		# 	gis_track = sector.add_track((min_r_pos-4, min_r_pos), r_pad_ratio=0.1)
		# 	# f_gis_track = sector.add_track((min_r_pos-3, min_r_pos), r_pad_ratio=0.1)
		# 	# r_gis_track = sector.add_track((min_r_pos-3, min_r_pos), r_pad_ratio=0.1)
		# 	min_r_pos -= 8
		# 	for gi_id in genomic_islands.keys():
		# 		start_locus = genomic_islands[gi_id][0]
		# 		end_locus = genomic_islands[gi_id][1]
		# 		if start_locus > end_locus:
		# 			start_gi = end_locus
		# 			end_gi = start_locus
		# 		else:
		# 			start_gi = start_locus
		# 			end_gi = end_locus

		# 		gis_track.rect(start_gi, end_gi, color=color)
		# 		label_pos = (start_gi + end_gi) / 2
		# 		gis_track.annotate(label_pos, f'{gi_id}', label_size=9)
		# 	print(f'added GIs track')



		# # Get info about genes
		# outf = open(outfilename, 'w')
		# labels, label_pos_list = [], []
		# features_sorted = dict(sorted(features.items()))
		# for feature in features_sorted.values():
		# # for feature in features:
		# 	start = int(feature.location.start)
		# 	end = int(feature.location.end)
		# 	label_pos = (start + end) / 2
		# 	gene_id = feature.qualifiers.get("gene_id", [None])[0]
		# 	label = feature.qualifiers.get("gene_name", [None])[0]
		# 	strand = feature.qualifiers.get("strand", [None])[0]
		# 	gene_type = feature.qualifiers.get("gene_type", [None])[0]
		# 	if gene_type == 'protein_coding':
		# 		function = feature.qualifiers.get("function", [None])[0]
		# 		outf.write(f'{gene_id}\t{strand}\t{start}\t{end}\t{feature.qualifiers.get("gene_name", [None])[0]}\t{feature.qualifiers.get("gene_type", [None])[0]}\t{function}\t{genes_of_interest_count[gene_id]}\t{statistics.mean(genes_of_interest_stat[gene_id])}\n')
		# 	else:
		# 		outf.write(f'{gene_id}\t{strand}\t{start}\t{end}\t{feature.qualifiers.get("gene_name", [None])[0]}\t{feature.qualifiers.get("gene_type", [None])[0]}\t{genes_of_interest_count[gene_id]}\t{statistics.mean(genes_of_interest_stat[gene_id])}\n')

		# # 	if label == None:
		# # 		continue
		# # 	if gene_id is not None:
		# # 		labels.append(gene_id)
		# # 		label_pos_list.append(label_pos)
		# # 	cds_track.annotate(label_pos, label, label_size=7)
		# outf.close()

	# Blast genome comparison & plot match blocks
	comp_name2color = {}
	# colors = ["black", "gray"]
	# run blast using pygenomeviz
	# align_coords = Blast([query_fasta, ref_fasta]).run()
	# align_coords = AlignCoord.filter(align_coords, identity_thr=MIN_IDENTITY)
	# run blast 		
	RunBlast(args, os.path.join(args.output_dir, 'blast', 'test_train_genomes'), testing_fasta, subject=[training_fasta], outfilename=f'{args.output_dir}/blast/test_train_genomes/test_train_genomes_blastn.out')
	align_coords = GetMatchRegions(args, f'{args.output_dir}/blast/test_train_genomes/test_train_genomes_blastn.out', genomic_islands, identity_thr=MIN_IDENTITY)

	# count the number of identical positions across the aligned regions
	identical_positions = 0
	# store percentage identity between matching regions
	percent_identity = []
	for sector in circos.sectors:
		blast_track = sector.add_track((min_r_pos-5, min_r_pos), r_pad_ratio=0.1)
		min_r_pos-5	
		for ac in align_coords:
			# # percent_identity.append(ac.identity)
			# # track = circos.get_sector(ac.query_name).tracks[-1] # Last added track in sector
			# # rect_color = interpolate_color("black", v=ac.identity, vmin=MIN_IDENTITY) # type: ignore
			percent_identity.append(ac[2])
			identical_positions += (ac[2]/100*(ac[1]-ac[0]))
			# print(f'{ac[2]}\t{ac[0]}\t{ac[1]}\t{(ac[1]-ac[0])}\t{ac[2]/100*(ac[1]-ac[0])}')
			rect_color = interpolate_color("black", v=ac[2], vmin=MIN_IDENTITY)
			blast_track.rect(ac[0], ac[1], color=rect_color)
			# # blast_track.rect(ac.query_start, ac.query_end, color=rect_color)
			# # matching_regions.append([ac.query_start, ac.query_end, ac.identity])

	# get stats on percentage identity
	avg_pct_identity = round(identical_positions/query_fasta.full_genome_length*100,2)
	ani = round(statistics.mean(percent_identity), 2)
	with open(os.path.join(args.output_dir, f'{args.label}_pct_identity_matching_regions.tsv'), 'w') as f:
		f.write(f'# identical positions\t{identical_positions}\npercentage identity\t{avg_pct_identity}%\n')
		f.write(f'Stats on aligned regions\nmean\t{statistics.mean(percent_identity)}\nmedian\t{statistics.median(percent_identity)}\nmin\t{min(percent_identity)}\nmax\t{max(percent_identity)}')



	for sector in circos.sectors:
		# define x-axis vector for the next tracks
		genome_pos = list(range(query_fasta.full_genome_length))

		# add track for scores
		min_r_pos -= 5
		scores_track = sector.add_track((min_r_pos-10, min_r_pos), r_pad_ratio=0.1)
		scores_track.axis(ec="deeppink")		
		y_values = list(range(math.floor(min(scores)), math.ceil(max(scores))+1, 1))
		y_labels = list(map(str, y_values))
		scores_track.yticks(y_values, y_labels)
		scores_track.line(genome_pos, scores, color="deeppink")
		print(f'added FN rate track')

		# add track for TP reads
		min_r_pos -= 13
		tp_track = sector.add_track((min_r_pos-10, min_r_pos), r_pad_ratio=0.1)
		tp_track.axis(ec="blue")
		pos_tp_count = [0]*query_fasta.full_genome_length
		for readid, data in tp_alignments_pos_test.items():
			for pos in range(data[2], data[3]+1, 1):
				pos_tp_count[pos-1] +=1
		y_values = list(range(min(pos_tp_count), max(pos_tp_count), 5))
		y_labels = list(map(str, y_values))
		tp_track.yticks(y_values, y_labels)
		tp_track.line(genome_pos, pos_tp_count, color="blue")
			# tp_track.rect(data[1], data[2], color="orange", lw=0.1)
		print(f'added TP track')

		# add tracks for FN reads 
		min_r_pos -= 13
		fn_track = sector.add_track((min_r_pos-10, min_r_pos), r_pad_ratio=0.1)
		fn_track.axis(ec="darkviolet")
		pos_fn_count = [0]*query_fasta.full_genome_length
		for readid, data in fn_alignments_pos_test.items():
			# if readid not in most_mapped_reads_id:
			for pos in range(data[2], data[3]+1, 1):
				pos_fn_count[pos-1] +=1
		y_values = list(range(min(pos_fn_count), max(pos_fn_count), 2))
		y_labels = list(map(str, y_values))
		fn_track.yticks(y_values, y_labels)
		fn_track.line(genome_pos, pos_fn_count, color="darkviolet")
		print(f'added FN track')

		# Plot GC skew
		min_r_pos -= 11
		gcskew_track = sector.add_track((min_r_pos-5, min_r_pos))
		pos_list, gcskews = GetGCSkew(test_record_seq)
		positive_gcskews = np.where(gcskews > 0, gcskews, 0)
		negative_gcskews = np.where(gcskews < 0, gcskews, 0)
		abs_max_gcskew = np.max(np.abs(gcskews))
		vmin, vmax = -abs_max_gcskew, abs_max_gcskew
		gcskew_track.fill_between(
			pos_list, positive_gcskews, 0, vmin=vmin, vmax=vmax, color="blue"
		)
		gcskew_track.fill_between(
			pos_list, negative_gcskews, 0, vmin=vmin, vmax=vmax, color="gold"
		)

		# Plot GC content
		min_r_pos -= 5
		gc_content_track = sector.add_track((min_r_pos-5, min_r_pos))
		pos_list, gc_content, test_genome_gc_content = GetGCContent(test_record_seq)
		gc_content_updated = gc_content - test_genome_gc_content
		positive_gc_content = np.where(gc_content_updated > 0, gc_content_updated, 0)
		negative_gc_content = np.where(gc_content_updated < 0, gc_content_updated, 0)
		abs_max_gc_content = np.max(np.abs(gc_content_updated))
		vmin, vmax = -abs_max_gc_content, abs_max_gc_content
		gc_content_track.fill_between(
			pos_list, positive_gc_content, 0, vmin=vmin, vmax=vmax, color="darkviolet"
		)
		gc_content_track.fill_between(
			pos_list, negative_gc_content, 0, vmin=vmin, vmax=vmax, color="orangered"
		)
		
		# report GC content of train and test genomes
		_, _, train_genome_gc_content = GetGCContent(train_record_seq)
		with open(os.path.join(args.output_dir, f'{args.label}_GC_content.tsv'), 'w') as f:
			f.write(f'Testing genome:\t{test_genome_gc_content}')
			f.write(f'Training genome:\t{train_genome_gc_content}')

		# get average GC content for FN and TP reads
		GetReadsGCcontent(args, gc_content, pos_list, fn_alignments_pos_test, 'fn')
		GetReadsGCcontent(args, gc_content, pos_list, tp_alignments_pos_test, 'tp')
		GetReadsGCcontent(args, gc_content_updated, pos_list, fn_alignments_pos_test, 'fn_relative')
		GetReadsGCcontent(args, gc_content_updated, pos_list, tp_alignments_pos_test, 'tp_relative')

	# Save figure
	# Enable annotation text adjustment (Default)
	# config.ann_adjust.enable = True
	fig = circos.plotfig()
	# Add legend
	handles = []
	if genomic_islands:
		handles.append(Patch(color='red', label='Genomic Islands'))
	handles += [
		Patch(color='black', label=f'{train_strain}\n{ref_fasta.full_genome_length:,} bp (training genome) - {avg_pct_identity}% - {ani}%'),
		Patch(color='deeppink', label='False Negative rate'),
		Patch(color='blue', label='True Positives'),
		Patch(color='darkviolet', label='False Negatives'),
		Line2D([], [], color='blue', label='Positive GC Skew', marker="^", ms=6, ls="None"),
		Line2D([], [], color='gold', label='Negative GC Skew', marker="v", ms=6, ls="None"),
		Line2D([], [], color='darkviolet', label='Positive GC Content', marker="^", ms=6, ls="None"),
		Line2D([], [], color='orangered', label='Negative GC Content', marker="v", ms=6, ls="None"),
		]
	_ = circos.ax.legend(handles=handles, bbox_to_anchor=(0.5, 0.475), loc="center", fontsize=8)

	fig.savefig(outfigpath, dpi=300)

	return avg_pct_identity, ani, test_strain, train_strain


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--training_fasta', type=str, help='path to file containing list of fasta files')
	parser.add_argument('--testing_fasta', type=str, help='path to file containing path to fasta files of testing genomes')
	parser.add_argument('--testing_genome', type=str, help='accession id of testing genome')
	parser.add_argument('--annotations_dir', type=str, help='path to directory containing gtf annotations files')
	parser.add_argument('--testing_fna_file', type=str, help='path to fasta file containing testing reads')
	# parser.add_argument('--training_fna_file', type=str, help='path to fasta file containing all training reads (label 1 and 0)')
	parser.add_argument('--label', type=str, help='label of species investigated', required=True)
	parser.add_argument('--sequences_info', type=str, help='path to file mapping labels of species in model to sequences id of all sequences in training set')
	parser.add_argument('--prob_threshold', type=float, help='probability score threshold', required=True)
	parser.add_argument('--rank', type=str, help='taxonomic rank investigated', choices=['species','genus','family','order','class', 'phylum'])
	parser.add_argument('--testing_results', type=str, help='path to file containing testing results')
	parser.add_argument('--genomic_islands', type=str, help='path to file containing list of genomic islands')
	parser.add_argument('--num_processes', type=int, help='number of processes to run in parallel')
	parser.add_argument('--output_dir', type=str, help='path to output directory')
	args = parser.parse_args()

	input_dir = os.getcwd()
	
	# get dltoda taxonomy
	path_dl_toda_tax = '/'.join(
                os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/data/dl_toda_taxonomy.tsv'
	with open(path_dl_toda_tax, 'r') as in_f:
		content = in_f.readlines()
		args.dl_toda_tax = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[1] for line in content}
	print(args.dl_toda_tax[args.label])
	# # retrieve accession and fasta files of testing and training genomes associated with each label
	# with open(args.testing_fasta, 'r') as f:
	# 	content = f.readlines()
	# 	args.test_genomes_info = {line.rstrip().split('\t')[0]: [line.rstrip().split('\t')[1], line.rstrip().split('\t')[2]] for line in content}

	with open(args.training_fasta, 'r') as f:
		content = f.readlines()
		args.train_genomes_info = {line.rstrip().split('\t')[0]: [line.rstrip().split('\t')[1], line.rstrip().split('\t')[2]] for line in content}

	# verify that the genomes investigated only have one chromosome
	testing_fasta, testing_records, training_fasta, training_records = CheckGenomes(args)
	
	# create output directories
	if not os.path.isdir(args.output_dir):
		os.makedirs(args.output_dir)
	if not os.path.isdir(os.path.join(args.output_dir, 'blast')):
		os.makedirs(os.path.join(args.output_dir, 'blast'))
	# if not os.path.isdir(os.path.join(args.output_dir, 'fp_analysis')):
	# 	os.makedirs(os.path.join(args.output_dir, 'fp_analysis'))
	if not os.path.isdir(os.path.join(args.output_dir, 'Genomes_GTF_missing')):
		os.makedirs(os.path.join(args.output_dir, 'Genomes_GTF_missing'))

	outfile_sum = open(os.path.join(args.output_dir, f'{args.label}_summary.tsv'), 'w')

	# get reads in testing set fasta file
	test_readid_to_read, test_sequence_length, test_ordered_reads_id = LoadFnaFile(args.testing_fna_file)

	# get FN and TP sequences
	fn_sequences = set()
	tp_sequences = set()
	fp_sequences = set()
	fn_cs = defaultdict(list)
	tp_cs = defaultdict(list)
	fp_cs = defaultdict(list)

	with open(args.testing_results, 'r') as f:
		for count, line in enumerate(f):
			prob = float(line.rstrip().split('\t')[2])
			if prob >= args.prob_threshold:
				if line.rstrip().split('\t')[0] == '1' and line.rstrip().split('\t')[1] == '0':
					fn_sequences.add(test_ordered_reads_id[count])
					fn_cs[test_ordered_reads_id[count]] = prob
				if line.rstrip().split('\t')[0] == '0' and line.rstrip().split('\t')[1] == '1':
					fp_sequences.add(test_ordered_reads_id[count])
					fp_cs[test_ordered_reads_id[count]] = prob
				if line.rstrip().split('\t')[0] == '1' and line.rstrip().split('\t')[1] == '1':
					tp_sequences.add(test_ordered_reads_id[count])
					tp_cs[test_ordered_reads_id[count]] = prob

	print(f'#FN for label {args.label}: {len(fn_sequences)}')
	print(f'#TP for label {args.label}: {len(tp_sequences)}')
	print(f'#FP for label {args.label}: {len(fp_sequences)}')
	outfile_sum.write(f'{len(fn_sequences)}\t{len(fp_sequences)}\t{len(tp_sequences)}\n')
	GetSeqLength(args, list(fn_sequences), test_sequence_length, 'fn')
	GetSeqLength(args, list(tp_sequences), test_sequence_length, 'tp')
	GetSeqLength(args, list(fp_sequences), test_sequence_length, 'fp')
	StoreCS(args, fn_cs, 'fp')
	StoreCS(args, tp_cs, 'tp')
	StoreCS(args, fp_cs, 'fp')

	# get association between sequences in training set and labels
	with open(args.sequences_info, 'r') as f:
		content = f.readlines()
		seq_to_labels = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[1] for line in content}

	# do FN analysis
	# create fasta file with testing reads from label 1
	with open(os.path.join(args.output_dir, f'{args.label}_test_reads.fna'), 'w') as outf:
		for k, v in test_readid_to_read.items():
			if k in fn_sequences or k in tp_sequences:
				outf.write(f'>{k}\n{v}\n')

	# blast testing reads to testing genome		
	RunBlast(args, os.path.join(args.output_dir, 'blast', 'test_reads_test_genome'), os.path.join(args.output_dir, f'{args.label}_test_reads.fna'), subject=[testing_fasta], outfilename=f'{args.output_dir}/blast/test_reads_test_genome/all_test_pos_test_blastn.out')
	# get mapping of false negatives to testing genome from label 1
	fn_alignments_pos_test = GetReadsAlignments(fn_sequences, f'{args.output_dir}/blast/test_reads_test_genome/all_test_pos_test_blastn.out', test_sequence_length, seq_to_labels, os.path.join(args.output_dir, f'blast/test_reads_test_genome/fn_pos_test_pos_test_{args.prob_threshold}_mapping_info.tsv'))

	# # blast testing reads to training genomes from other species
	# training_genomes = [v[1] for k, v in args.train_genomes_info.items() if k != args.label]
	# RunBlast(args, os.path.join(args.output_dir, 'blast'), os.path.join(args.output_dir, f'{args.label}_test_reads.fna'), subject=training_genomes, outfilename=f'{args.output_dir}/blast/all_test_pos_train_blastn.out')
	# fn_alignments_pos_neg_train = GetReadsAlignments(fn_sequences, f'{args.output_dir}/blast/all_test_pos_train_blastn.out', test_sequence_length, seq_to_labels, os.path.join(args.output_dir, f'fn_pos_test_neg_train_{args.prob_threshold}_mapping_info.tsv'))
	# # get taxonomy of mapped training genomes and taxon with most reads mapped
	# _ = GetFNOtherInfo(args, fn_alignments_pos_test, fn_alignments_pos_neg_train, pos_test_annot_info, test_sequence_length, test_readid_to_read)
	
	# do TP analysis
	# get mapping of true positives to testing genome from label 1
	tp_alignments_pos_test = GetReadsAlignments(tp_sequences, f'{args.output_dir}/blast/test_reads_test_genome/all_test_pos_test_blastn.out', test_sequence_length, seq_to_labels, os.path.join(args.output_dir, f'blast/test_reads_test_genome/tp_pos_test_pos_test_{args.prob_threshold}_mapping_info.tsv'))
	# _ = GetGenes(args, args.label, args.output_dir, pos_test_annot_info, tp_alignments_pos_test, test_sequence_length, test_readid_to_read, 'TP')

	# get false negative or false positive rate
	# scores, fn_reads_kept, tp_reads_kept = GetScores(args, testing_records, tp_alignments_pos_test, fn_alignments_pos_test)

	# get annotations info
	pos_test_annot_info, _ = GetAnnotInfo(args, args.testing_genome, input_dir)
	scores, fn_genes, tp_genes = GetGenes(args, pos_test_annot_info, fn_alignments_pos_test, tp_alignments_pos_test, test_sequence_length, test_readid_to_read, len(testing_records[0].seq), fn_cs, tp_cs)
	
	# store FN and TP reads in tsv files for analysis of the attentions weights
	CreateTsvFile(tp_sequences, test_readid_to_read, os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_tp_reads_all.tsv'))
	CreateTsvFile(fn_sequences, test_readid_to_read, os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_fn_reads_all.tsv'))
	# GetReadsForAttentions(args, tp_alignments_pos_test, fn_alignments_pos_test, test_readid_to_read)

	# get info about genomic islands and/or create Circos plot
	if args.genomic_islands is not None:
		if os.path.isdir(args.genomic_islands):
			gis_align = GetGIsFromFasta(args, args.testing_genome, testing_fasta)
		else:
			gis_align = GetGIsFromAnnotations(args, input_dir, str(training_records[0].seq), args.train_genomes_info[args.label][0], testing_fasta)

		avg_pct_identity, ani, test_strain, train_strain = CircosPlot(args, scores, testing_records[0].seq, training_records[0].seq, testing_fasta, training_fasta, fn_alignments_pos_test, tp_alignments_pos_test, \
			os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_fn_circos.png'), genomic_islands=gis_align)
	else:
		avg_pct_identity, ani, test_strain, train_strain = CircosPlot(args, scores, testing_records[0].seq, training_records[0].seq, testing_fasta, training_fasta, fn_alignments_pos_test, tp_alignments_pos_test, \
			os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_fn_circos.png'))
	
	species = args.dl_toda_tax[args.label].split(';')[0]
	genus = args.dl_toda_tax[args.label].split(';')[1]
	with open(os.path.join(args.output_dir, f'{args.label}_fn_unique_genes_{args.prob_threshold}.tsv'), 'w') as f:
		for k, v in fn_genes.items():
			f.write(f'{args.label}\t1\t{args.testing_genome}\t{test_strain}\t{args.train_genomes_info[args.label][0]}\t{train_strain}\t{species}\t{genus}\t{avg_pct_identity}\t{ani}\t{k}\t{v[0]}\t')
			if pos_test_annot_info[k][0] == 'protein_coding':
				f.write(f'{pos_test_annot_info[k][0]}\t{pos_test_annot_info[k][4]}\t{pos_test_annot_info[k][5]}\n')
			else:
				f.write(f'{pos_test_annot_info[k][0]}\t{pos_test_annot_info[k][4]}\tNA\n')


	with open(os.path.join(args.output_dir, f'{args.label}_tp_shared_genes_{args.prob_threshold}.tsv'), 'w') as f:
		for k, v in tp_genes.items():
			f.write(f'{args.label}\t1\t{args.testing_genome}\t{test_strain}\t{args.train_genomes_info[args.label][0]}\t{train_strain}\t{species}\t{genus}\t{avg_pct_identity}\t{ani}\t{k}\t{v[0]}\t')
			if pos_test_annot_info[k][0] == 'protein_coding':
				f.write(f'{pos_test_annot_info[k][0]}\t{pos_test_annot_info[k][4]}\t{pos_test_annot_info[k][5]}\n')
			else:
				f.write(f'{pos_test_annot_info[k][0]}\t{pos_test_annot_info[k][4]}\tNA\n')

	# get taxa of FP reads
	if len(fp_sequences) > 0:
		fp_labels = set([s.split('|')[1] for s in list(fp_sequences)])
		fp_taxa = defaultdict(int)
		for label in fp_labels:
			# get fp sequences of label
			label_sequences = set([seq_id for seq_id in fp_sequences if seq_id.split('|')[1] == label])
			# monitor number of sequences per label
			fp_taxa[label] = len(label_sequences)
		fp_taxa_sorted = dict(sorted(fp_taxa.items(), key=lambda item: item[1], reverse=True))
		with open(os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_fp_neg_taxa.tsv'), 'w') as f:
			for k, v in fp_taxa_sorted.items():
				f.write(f'{k}\t{args.dl_toda_tax[k]}\t{v}\n')
		




