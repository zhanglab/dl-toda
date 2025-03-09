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
	with open(args.test_genomes_info[args.label][1], "r") as handle:
		test_records = list(SeqIO.parse(handle, "fasta"))

	# load training fasta file
	with open(args.train_genomes_info[args.label][1], "r") as handle:
		train_records = list(SeqIO.parse(handle, "fasta"))

	assert len(test_records) == 1, f'{arg.label}\t{args.test_genomes_info[args.label][0]} has more than 1 chromosome'
	assert len(train_records) == 1, f'{arg.label}\t{args.train_genomes_info[args.label][0]} has more than 1 chromosome'

	return args.test_genomes_info[args.label][1], test_records, args.train_genomes_info[args.label][1], train_records


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


def CreateFastaFile(genes_of_interest, alignments, readid_to_read, filename):
	reads_of_interest = set()
	for readid, data in alignments.items():
		start_pos = data[2]
		end_pos = data[3]
		for gene_id, annot in genes_of_interest.items():
			if (start_pos <= annot[1] and end_pos >= annot[2]) or \
			(start_pos <= annot[1] and end_pos >= annot[1]) or \
			(start_pos >= annot[1] and end_pos <= annot[2]) or \
			(start_pos <= annot[2] and end_pos >= annot[2]):
				reads_of_interest.add(readid)
	
	with open(filename, 'w') as f:
		f.write(''.join([f'>{r}\n{readid_to_read[r]}\n' for r in list(reads_of_interest)]))


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



def GetGenes(args, label, output_dir, annot_info, alignments, sequence_length, readid_to_read, type):
	# get length and function of fn sequences per mapped position on the genome investigated
	genes = defaultdict(list)
	functions = defaultdict(int)
	genestype = defaultdict(int)
	readid_w_gene = defaultdict(list)
	pos_readid = defaultdict(list) # key: position in target genome, value: list of reads id mapped to that position

	for readid, data in alignments.items():
		if data[2] < data[3]:
			start_pos = data[2]
			end_pos = data[3]
		else:
			start_pos = data[3]
			end_pos = data[2]
		for pos in range(start_pos, end_pos+1, 1):
			pos_readid[pos-1].append(readid)
		for gene_id, annot in annot_info.items():
			if (start_pos <= annot[1] and end_pos >= annot[2]) or \
			(start_pos <= annot[1] and end_pos >= annot[1]) or \
			(start_pos >= annot[1] and end_pos <= annot[2]) or \
			(start_pos <= annot[2] and end_pos >= annot[2]):
				if annot[0] == 'protein_coding':
					functions[annot[5]] += 1
				genes[gene_id] = annot
				readid_w_gene[readid] = [data[2], data[3], gene_id]
				genestype[annot[0]] += 1

	pos_readid_count = [len(v) for v in pos_readid.values()]
	
	if len(pos_readid_count) > 0:
		print(f'mean: {statistics.mean(pos_readid_count)}\tmedian: {statistics.median(pos_readid_count)}\tmin: {min(pos_readid_count)}\tmax: {max(pos_readid_count)}')

	genes_of_interest = defaultdict(list)
	for pos, list_readid in pos_readid.items():
		if len(list_readid) >= 3:
			for readid in list_readid:
				if readid in readid_w_gene:
					gene_id = readid_w_gene[readid][2]
					genes_of_interest[gene_id] = genes[gene_id]

	reads_wo_genes = []
	if len(readid_w_gene) != len(alignments):
		with open(os.path.join(output_dir, f'{label}_{type}_reads_wo_gene_{args.prob_threshold}.tsv'), 'w') as f:
			for readid, data in alignments.items():
				if readid not in readid_w_gene:
					f.write(f'{readid}\t{sequence_length[readid]}\t{data[0]}\t{data[1]}\t{data[2]}\t{data[3]}\n')
					reads_wo_genes.append(readid)

		with open(os.path.join(output_dir, f'{label}_{type}_wo_gene_{args.prob_threshold}.fna'), 'w') as f:
			f.write(''.join([f'>{r}\n{readid_to_read[r]}\n' for r in reads_wo_genes]))
	else:
		print('all reads were found a gene')

	with open(os.path.join(output_dir, f'{label}_{type}_genes_info_{args.prob_threshold}.tsv'), 'w') as outf:
		for gene_id, annot in genes_of_interest.items():
			if annot[0] == 'protein_coding':
				outf.write(f'{gene_id}\t{annot[3]}\t{annot[1]}\t{annot[2]}\t{annot[4]}\t{annot[0]}\t{annot[5]}\n')
			else:
				outf.write(f'{gene_id}\t{annot[3]}\t{annot[1]}\t{annot[2]}\t{annot[4]}\t{annot[0]}\n')

	functions_sorted = dict(sorted(functions.items(), key=lambda item: item[1], reverse=True))
	with open(os.path.join(output_dir, f'{label}_{type}_functions_{args.prob_threshold}.tsv'), 'w') as f:
		for k, v in functions_sorted.items():
			f.write(f'{k}\t{v}\n')

	genestype_sorted = dict(sorted(genestype.items(), key=lambda item: item[1], reverse=True))
	with open(os.path.join(output_dir, f'{label}_{type}_genes_type_{args.prob_threshold}.tsv'), 'w') as f:
		f.write(f'{genestype["protein_coding"]}\t{genestype["tRNA"]}\t{genestype["rRNA"]}\n')

	return genes_of_interest


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


def GetShanningScore(testing_records, tp_alignments, fn_alignments):
	genome_size = len(testing_records[0].seq)
	shannon_scores = []
	for i in range(1, genome_size+1, 1):
		num_tp = 0
		num_fn = 0
		
		# check if position is located in a read assigned to TP
		for read_id, data in tp_alignments.items():
			if i >= data[2] and i <= data[3]:
				num_tp += 1

		# check if position is located in a read assigned to FN
		for read_id, data in fn_alignments.items():
			if i >= data[2] and i <= data[3]:
				num_fn += 1

		# compute probability for each group
		if num_tp+num_fn > 0:
			prob_tp = num_tp / (num_tp+num_fn)
			prob_fn = num_fn / (num_tp+num_fn)

			# compute tp and fn contribution to shannon score
			shannon_tp = prob_tp*math.log(prob_tp, 2) if prob_tp > 0 else 0
			shannon_fn = prob_tp*math.log(prob_fn, 2) if prob_fn > 0 else 0

			# compute shannon entropy
			if (shannon_tp + shannon_fn) == 0:
				print('equal to 0', num_tp, prob_tp, shannon_tp, num_fn, prob_fn, shannon_fn, shannon_entropy)
				shannon_entropy = 0
			else:
				shannon_entropy = -(shannon_tp + shannon_fn)
				print(' NOT equal to 0', num_tp, prob_tp, shannon_tp, num_fn, prob_fn, shannon_fn, shannon_entropy)
		else:
			shannon_entropy = 0

		shannon_scores.append(shannon_entropy)

	print(f'shannon entropy:\nmean\t{statistics.mean(shannon_scores)}\nmedian\t{statistics.median(shannon_scores)}\nmin\t{min(shannon_scores)}\nmax\t{max(shannon_scores)}')
	return shannon_scores


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


def StoreCS(args, list_cs, type):
	with open(os.path.join(args.output_dir, f'{args.label}_{type}_{args.prob_threshold}.tsv'), 'w') as f:
		f.write('\n'.join([str(x) for x in list_cs]))


def GetSeqLength(args, sequences_id, sequence_length, type):
	if len(sequences_id) != 0:
		seq_length_info = [sequence_length[s] for s in sequences_id]
		print(f'{type}\tmean: {statistics.mean(seq_length_info)}\tmedian: {statistics.median(seq_length_info)}\tmax: {max(seq_length_info)}\tmin: {min(seq_length_info)}')

		with open(os.path.join(args.output_dir, f'{args.label}_{type}_{args.prob_threshold}_seq_length.tsv'), 'w') as f:
			f.write('\n'.join([str(x) for x in seq_length_info]))

			# f.write(f'{statistics.mean(seq_length_info)}\t{statistics.median(seq_length_info)}\t{max(seq_length_info)}\t{min(seq_length_info)}')


def FNCircosPlot(args, test_record_seq, train_record_seq, testing_fasta, training_fasta, \
			fn_alignments_pos_test, tp_alignments_pos_test, genes_of_interest, outfigpath, \
			outfilename, shannon_scores, genomic_islands=None):
	
	# load data from training and testing genomes of label 1
	query_fasta = Fasta(testing_fasta) # query --> testing genome
	ref_fasta = Fasta(training_fasta) # ref/subject --> training genome

	# Initialize circos instance
	circos = Circos(
	    sectors=query_fasta.get_seqid2size(),
	    # space=0 if len(ref_fasta.get_seqid2size()) == 1 else 2,
		space=10,
	)
	with open(testing_fasta, 'r') as f:
		content = f.readline()
	test_strain = ' '.join(content.split(',')[0].split(' ')[1:])
	circos.text(f'{test_strain}\n{query_fasta.full_genome_length:,} bp\n(testing genome)', size=11, r=20)

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
		if query_fasta.full_genome_length > 4000000:
			outer_track.xticks_by_interval(TICKS_INTERVAL, label_formatter=lambda v: f"{v/1000000:.1f} Mb", outer=False,)
			outer_track.xticks_by_interval(100000, tick_length=1, show_label=False)
		if query_fasta.full_genome_length < 2000000:
			outer_track.xticks_by_interval(TICKS_INTERVAL, label_formatter=lambda v: f"{v/500000:.1f} Mb", outer=False,)
			outer_track.xticks_by_interval(100000, tick_length=1, show_label=False)
		min_r_pos -= 6

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
		features = {}
		for gene_id in genes_of_interest.keys():
			if genes_of_interest[gene_id][3] == '+':
				location = FeatureLocation(start=genes_of_interest[gene_id][1], end=genes_of_interest[gene_id][2], strand=+1)
				if genes_of_interest[gene_id][0] == 'protein_coding':
					feature = SeqFeature(location=location, qualifiers={"gene_type": [genes_of_interest[gene_id][0]], "gene_id": [gene_id], "gene_name": [genes_of_interest[gene_id][4]], "strand": ["plus"], "function": [genes_of_interest[gene_id][5]]})
					# cds_track.genomic_features(feature, plotstyle="arrow", fc="red")
				else:
					feature = SeqFeature(location=location, qualifiers={"gene_type": [genes_of_interest[gene_id][0]], "gene_id": [gene_id], "gene_name": [genes_of_interest[gene_id][4]], "strand": ["plus"]})
					# if genes_of_interest[gene_id][0] == 'tRNA':
						# cds_track.genomic_features(feature, fc="darkgreen")
					# if genes_of_interest[gene_id][0] == 'rRNA':
					# 	rrna_track.genomic_features(feature, fc="deeppink")
			else:
				location = FeatureLocation(start=genes_of_interest[gene_id][1], end=genes_of_interest[gene_id][2], strand=-1)
				if genes_of_interest[gene_id][0] == 'protein_coding':
					feature = SeqFeature(location=location, qualifiers={"gene_type": [genes_of_interest[gene_id][0]], "gene_id": [gene_id], "gene_name": [genes_of_interest[gene_id][4]], "strand": ["minus"], "function": [genes_of_interest[gene_id][5]]})
					# cds_track.genomic_features(feature, plotstyle="arrow", fc="blue")
				else:
					feature = SeqFeature(location=location, qualifiers={"gene_type": [genes_of_interest[gene_id][0]], "gene_id": [gene_id], "gene_name": [genes_of_interest[gene_id][4]], "strand": ["minus"]})
					# if genes_of_interest[gene_id][0] == 'tRNA':
						# cds_track.genomic_features(feature, fc="darkgreen")
					# if genes_of_interest[gene_id][0] == 'rRNA':
					# 	rrna_track.genomic_features(feature, fc="deeppink")

			# features.append(feature)
			features[genes_of_interest[gene_id][1]] = feature

		
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



		# Get info about genes
		outf = open(outfilename, 'w')
		labels, label_pos_list = [], []
		features_sorted = dict(sorted(features.items()))
		for feature in features_sorted.values():
		# for feature in features:
			start = int(feature.location.start)
			end = int(feature.location.end)
			label_pos = (start + end) / 2
			gene_id = feature.qualifiers.get("gene_id", [None])[0]
			label = feature.qualifiers.get("gene_name", [None])[0]
			strand = feature.qualifiers.get("strand", [None])[0]
			gene_type = feature.qualifiers.get("gene_type", [None])[0]
			if gene_type == 'protein_coding':
				function = feature.qualifiers.get("function", [None])[0]
				outf.write(f'{gene_id}\t{strand}\t{start}\t{end}\t{feature.qualifiers.get("gene_name", [None])[0]}\t{feature.qualifiers.get("gene_type", [None])[0]}\t{function}\n')
			else:
				outf.write(f'{gene_id}\t{strand}\t{start}\t{end}\t{feature.qualifiers.get("gene_name", [None])[0]}\t{feature.qualifiers.get("gene_type", [None])[0]}\n')

		# 	if label == None:
		# 		continue
		# 	if gene_id is not None:
		# 		labels.append(gene_id)
		# 		label_pos_list.append(label_pos)
		# 	cds_track.annotate(label_pos, label, label_size=7)
		outf.close()

	# Blast genome comparison & plot match blocks
	comp_name2color = {}
	# colors = ["black", "gray"]
	# store percentage identity between matching regions
	percent_identity = []
	# run blast using pygenomeviz
	# align_coords = Blast([query_fasta, ref_fasta]).run()
	# align_coords = AlignCoord.filter(align_coords, identity_thr=MIN_IDENTITY)
	# run blast 		
	RunBlast(args, os.path.join(args.output_dir, 'blast', 'test_train_genomes'), testing_fasta, subject=[training_fasta], outfilename=f'{args.output_dir}/blast/test_train_genomes/test_train_genomes_blastn.out')
	align_coords = GetMatchRegions(args, f'{args.output_dir}/blast/test_train_genomes/test_train_genomes_blastn.out', genomic_islands, identity_thr=MIN_IDENTITY)

	# color = ColorCycler()
	# comp_name2color[comp_fasta.name] = colors[idx]
	matching_regions = []
	for sector in circos.sectors:
		blast_track = sector.add_track((min_r_pos-5, min_r_pos), r_pad_ratio=0.1)
		min_r_pos-5	
		for ac in align_coords:
			print(ac)
			# # percent_identity.append(ac.identity)
			# # track = circos.get_sector(ac.query_name).tracks[-1] # Last added track in sector
			# # rect_color = interpolate_color("black", v=ac.identity, vmin=MIN_IDENTITY) # type: ignore
			percent_identity.append(ac[2])
			rect_color = interpolate_color("black", v=ac[2], vmin=MIN_IDENTITY)
			blast_track.rect(ac[0], ac[1], color=rect_color)
			matching_regions.append([ac[0], ac[1], ac[2]])
			# # blast_track.rect(ac.query_start, ac.query_end, color=rect_color)
			# # matching_regions.append([ac.query_start, ac.query_end, ac.identity])

	pos_matching_regions = set()
	for i in range(len(matching_regions)):
		for j in range(matching_regions[i][0], matching_regions[i][1]+1, 1):
			pos_matching_regions.add(j)

	pos_not_matching_regions = [i for i in range(1, query_fasta.full_genome_length+1, 1) if i not in pos_matching_regions]

	fn_matching_regions = set() # key = position on testing genome, value = 1 if mapped at least once by a false negative read
	for read_id, data in fn_alignments_pos_test.items():
		start_pos = data[2]
		end_pos = data[3]
		for i in range(len(matching_regions)):
			if (start_pos <= matching_regions[i][0] and end_pos >= matching_regions[i][1]) or \
				(start_pos >= matching_regions[i][0] and end_pos <= matching_regions[i][1]) or  \
				(start_pos <= matching_regions[i][0] and end_pos >= matching_regions[i][0]) or  \
				(start_pos <= matching_regions[i][1] and end_pos >= matching_regions[i][1]):
				fn_matching_regions.add(read_id)
	fn_not_matching_regions = [r for r in fn_alignments_pos_test.keys() if r not in fn_matching_regions]	

	tp_matching_regions = set()
	for read_id, data in tp_alignments_pos_test.items():
		start_pos = data[2]
		end_pos = data[3]
		for i in range(len(matching_regions)):
			if (start_pos <= matching_regions[i][0] and end_pos >= matching_regions[i][0]) or \
				(start_pos >= matching_regions[i][0] and end_pos <= matching_regions[i][1]) or \
				(start_pos <= matching_regions[i][0] and end_pos >= matching_regions[i][0]) or  \
				(start_pos <= matching_regions[i][1] and end_pos >= matching_regions[i][1]):
				tp_matching_regions.add(read_id)
	tp_not_matching_regions = [r for r in tp_alignments_pos_test.keys() if r not in tp_matching_regions]
			
	with open(os.path.join(args.output_dir, f'{args.label}_FN_TP_matching_regions.tsv'), 'w') as f:
		f.write(f'% testing genome that matches to training genome\t{len(pos_matching_regions)}\t{query_fasta.full_genome_length}\t{round(len(pos_matching_regions)/query_fasta.full_genome_length, 3)*100}')
		f.write(f'% testing genome that does not match to training genome\t{len(pos_not_matching_regions)}\t{query_fasta.full_genome_length}\t{round(len(pos_not_matching_regions)/query_fasta.full_genome_length, 3)*100}')
		f.write(f'% of FN reads mapped to matching regions\t{len(fn_matching_regions)}\t{len(fn_not_matching_regions)}\t{len(fn_alignments_pos_test)}\t{round(len(fn_matching_regions)/len(fn_sequences), 3)*100}')
		f.write(f'% of FN reads mapped to not matching regions\t{len(fn_matching_regions)}\t{len(fn_not_matching_regions)}\t{len(fn_alignments_pos_test)}\t{round(len(fn_not_matching_regions)/len(fn_sequences), 3)*100}')
		f.write(f'% of TP reads mapped to matching regions\t{len(tp_matching_regions)}\t{len(tp_not_matching_regions)}\t{len(tp_alignments_pos_test)}\t{round(len(tp_matching_regions)/len(tp_sequences), 3)*100}')
		f.write(f'% of TP reads mapped to not matching regions\t{len(tp_matching_regions)}\t{len(tp_not_matching_regions)}\t{len(tp_alignments_pos_test)}\t{round(len(tp_not_matching_regions)/len(tp_sequences), 3)*100}')

	# get stats on percentage identity
	with open(os.path.join(args.output_dir, f'{args.label}_pct_identity_matching_regions.tsv'), 'w') as f:
		f.write(f'{statistics.mean(percent_identity)}\t{statistics.median(percent_identity)}\t{min(percent_identity)}\t{max(percent_identity)}')

	for sector in circos.sectors:
		# define x-axis vector for the next tracks
		genome_pos = list(range(query_fasta.full_genome_length))

		# add track for shannon entropy scores
		min_r_pos -= 5
		scores_track = sector.add_track((min_r_pos-10, min_r_pos), r_pad_ratio=0.1)
		scores_track.axis(ec="darkorange")
		y_values = list(range(min(shannon_scores), max(shannon_scores), 1))
		y_labels = list(map(str, y_values))
		scores_track.yticks(y_values, y_labels)
		scores_track.line(genome_pos, shannon_scores, color="darkorange")
		print(f'added Scores track')

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
		min_r_pos -= 11
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
			pos_list, positive_gcskews, 0, vmin=vmin, vmax=vmax, color="grey"
		)
		gcskew_track.fill_between(
			pos_list, negative_gcskews, 0, vmin=vmin, vmax=vmax, color="limegreen"
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
			pos_list, positive_gc_content, 0, vmin=vmin, vmax=vmax, color="black"
		)
		gc_content_track.fill_between(
			pos_list, negative_gc_content, 0, vmin=vmin, vmax=vmax, color="deeppink"
		)
		
		# report GC content of train and test genomes
		_, _, train_genome_gc_content = GetGCContent(train_record_seq)
		with open(os.path.join(args.output_dir, f'{args.label}_GC_content.tsv'), 'w') as f:
			f.write(f'Testing genome:\t{test_genome_gc_content}')
			f.write(f'Training genome:\t{train_genome_gc_content}')

		# get average GC content for FN and TP reads
		GetReadsGCcontent(args, gc_content, pos_list, fn_alignments_pos_test, 'FN')
		GetReadsGCcontent(args, gc_content, pos_list, tp_alignments_pos_test, 'TP')
		GetReadsGCcontent(args, gc_content_updated, pos_list, fn_alignments_pos_test, 'FN_relative')
		GetReadsGCcontent(args, gc_content_updated, pos_list, tp_alignments_pos_test, 'TP_relative')

	# Save figure
	# Enable annotation text adjustment (Default)
	# config.ann_adjust.enable = True
	fig = circos.plotfig()
	# Add legend
	with open(training_fasta, 'r') as f:
		content = f.readline()
	train_strain = ' '.join(content.split(',')[0].split(' ')[1:])

	handles = []
	if genomic_islands:
		handles.append(Patch(color='red', label='Genomic Islands'))
	handles += [
		Patch(color='black', label=f'{train_strain}\n(training genome)'),
		Patch(color='darkorange', label='Shannon Entropy'),
		Patch(color='blue', label='True Positives'),
		Patch(color='darkviolet', label='False Negatives'),
		Line2D([], [], color='grey', label='Positive GC Skew', marker="^", ms=6, ls="None"),
		Line2D([], [], color='limegreen', label='Negative GC Skew', marker="v", ms=6, ls="None"),
		Line2D([], [], color='black', label='Positive GC Content', marker="^", ms=6, ls="None"),
		Line2D([], [], color='deeppink', label='Negative GC Content', marker="v", ms=6, ls="None"),
		]
	_ = circos.ax.legend(handles=handles, bbox_to_anchor=(0.5, 0.475), loc="center", fontsize=8)

	fig.savefig(outfigpath, dpi=300)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--training_fasta', type=str, help='path to file containing list of fasta files')
	parser.add_argument('--testing_fasta', type=str, help='path to file containing path to fasta files of training genomes')
	parser.add_argument('--annotations_dir', type=str, help='path to directory containing gtf annotations files')
	parser.add_argument('--testing_fna_file', type=str, help='path to fasta file containing all testing reads (label 1 and 0)')
	parser.add_argument('--training_fna_file', type=str, help='path to fasta file containing all training reads (label 1 and 0)')
	parser.add_argument('--label', type=str, help='label of species investigated', required=True)
	parser.add_argument('--sequences_info', type=str, help='path to file mapping labels of species in model to sequences id of all sequences in training set')
	parser.add_argument('--prob_threshold', type=float, help='probability score threshold', required=True)
	parser.add_argument('--rank', type=str, help='taxonomic rank investigated', choices=['species','genus','family','order','class', 'phylum'])
	parser.add_argument('--testing_results', type=str, help='path to file containing testing results')
	parser.add_argument('--genomic_islands', type=str, help='path to file containing list of genomic islands')
	parser.add_argument('--num_processes', type=int, help='number of processes to run in parallel')
	args = parser.parse_args()

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
	testing_fasta, testing_records, training_fasta, training_records = CheckGenomes(args)
	
	# create output directories
	args.output_dir = os.path.join(os.getcwd(), args.label)
	if not os.path.isdir(args.output_dir):
		os.makedirs(args.output_dir)
	if not os.path.isdir(os.path.join(args.output_dir, 'blast')):
		os.makedirs(os.path.join(args.output_dir, 'blast'))
	if not os.path.isdir(os.path.join(args.output_dir, 'FP_analysis')):
		os.makedirs(os.path.join(args.output_dir, 'FP_analysis'))
	if not os.path.isdir(os.path.join(args.output_dir, 'Genomes_GTF_missing')):
		os.makedirs(os.path.join(args.output_dir, 'Genomes_GTF_missing'))

	outfile_sum = open(os.path.join(args.output_dir, f'{args.label}_summary.tsv'), 'w')

	# get reads in testing set fasta file
	test_readid_to_read, test_sequence_length, test_ordered_reads_id = LoadFnaFile(args.testing_fna_file)

	# get FN and TP sequences
	fn_sequences = set()
	tp_sequences = set()
	fp_sequences = set()
	fn_cs = []
	tp_cs = []
	fp_cs = []

	with open(args.testing_results, 'r') as f:
		for count, line in enumerate(f):
			prob = float(line.rstrip().split('\t')[2])
			if prob >= args.prob_threshold:
				if line.rstrip().split('\t')[0] == '1' and line.rstrip().split('\t')[1] == '0':
					fn_sequences.add(test_ordered_reads_id[count])
					fn_cs.append(prob)
				if line.rstrip().split('\t')[0] == '0' and line.rstrip().split('\t')[1] == '1':
					fp_sequences.add(test_ordered_reads_id[count])
					fp_cs.append(prob)
				if line.rstrip().split('\t')[0] == '1' and line.rstrip().split('\t')[1] == '1':
					tp_sequences.add(test_ordered_reads_id[count])
					tp_cs.append(prob)

	print(f'#FN for label {args.label}: {len(fn_sequences)}')
	print(f'#TP for label {args.label}: {len(tp_sequences)}')
	print(f'#FP for label {args.label}: {len(fp_sequences)}')
	outfile_sum.write(f'{len(fn_sequences)}\t{len(fp_sequences)}\t{len(tp_sequences)}\n')
	GetSeqLength(args, list(fn_sequences), test_sequence_length, 'FN')
	GetSeqLength(args, list(tp_sequences), test_sequence_length, 'TP')
	GetSeqLength(args, list(fp_sequences), test_sequence_length, 'FP')
	StoreCS(args, fn_cs, 'FN')
	StoreCS(args, tp_cs, 'TP')
	StoreCS(args, fp_cs, 'FP')

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
	fn_alignments_pos_test = GetReadsAlignments(fn_sequences, f'{args.output_dir}/blast/test_reads_test_genome/all_test_pos_test_blastn.out', test_sequence_length, seq_to_labels, os.path.join(args.output_dir, f'blast/test_reads_test_genome/FN_pos_test_pos_test_{args.prob_threshold}_mapping_info.tsv'))

	# # blast testing reads to training genomes from other species
	# training_genomes = [v[1] for k, v in args.train_genomes_info.items() if k != args.label]
	# RunBlast(args, os.path.join(args.output_dir, 'blast'), os.path.join(args.output_dir, f'{args.label}_test_reads.fna'), subject=training_genomes, outfilename=f'{args.output_dir}/blast/all_test_pos_train_blastn.out')
	# fn_alignments_pos_neg_train = GetReadsAlignments(fn_sequences, f'{args.output_dir}/blast/all_test_pos_train_blastn.out', test_sequence_length, seq_to_labels, os.path.join(args.output_dir, f'FN_pos_test_neg_train_{args.prob_threshold}_mapping_info.tsv'))
	# # get taxonomy of mapped training genomes and taxon with most reads mapped
	# _ = GetFNOtherInfo(args, fn_alignments_pos_test, fn_alignments_pos_neg_train, pos_test_annot_info, test_sequence_length, test_readid_to_read)
	
	# do TP analysis
	# get mapping of true positives to testing genome from label 1
	tp_alignments_pos_test = GetReadsAlignments(tp_sequences, f'{args.output_dir}/blast/test_reads_test_genome/all_test_pos_test_blastn.out', test_sequence_length, seq_to_labels, os.path.join(args.output_dir, f'blast/test_reads_test_genome/TP_pos_test_pos_test_{args.prob_threshold}_mapping_info.tsv'))
	# _ = GetGenes(args, args.label, args.output_dir, pos_test_annot_info, tp_alignments_pos_test, test_sequence_length, test_readid_to_read, 'TP')

	# get shannon entropy scores
	shannon_scores = GetShanningScore(testing_records, tp_alignments_pos_test, fn_alignments_pos_test)

	# get annotations info
	pos_test_annot_info, _ = GetAnnotInfo(args, args.test_genomes_info[args.label][0], input_dir)
	fn_genes_of_interest = GetGenes(args, args.label, args.output_dir, pos_test_annot_info, fn_alignments_pos_test, test_sequence_length, test_readid_to_read, 'FN')

	# GetReadsForAttentions(args, tp_alignments_pos_test, fn_alignments_pos_test, test_readid_to_read)

	# # create fastq files with FN and TP reads mapping positions of interest on the testing genome
	# CreateFastaFile(fn_genes_of_interest, fn_alignments_pos_test, test_readid_to_read, os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_fn_reads.fq'))
	# CreateFastaFile(fn_genes_of_interest, tp_alignments_pos_test, test_readid_to_read, os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_tp_reads.fq'))

	# blast testing reads to training genome from label 1
	# RunBlast(args, os.path.join(args.output_dir, 'blast', 'test_reads_train_genome'), os.path.join(args.output_dir, f'{args.label}_test_reads.fna'), subject=[training_fasta], outfilename=f'{args.output_dir}/blast/test_reads_train_genome/all_test_pos_train_blastn.out')
	# test_alignments_pos_train = GetReadsAlignments(fn_sequences.union(tp_sequences), f'{args.output_dir}/blast/test_reads_train_genome/all_test_pos_train_blastn.out', test_sequence_length, seq_to_labels, os.path.join(args.output_dir, f'blast/test_reads_train_genome/pos_test_pos_train_{args.prob_threshold}_mapping_info.tsv'))
	
	# get info about genomic islands
	if args.genomic_islands is not None:
		if os.path.isdir(args.genomic_islands):
			gis_align = GetGIsFromFasta(args, args.test_genomes_info[args.label][0], testing_fasta)
		else:
			gis_align = GetGIsFromAnnotations(args, input_dir, str(training_records[0].seq), args.train_genomes_info[args.label][0], testing_fasta)

		FNCircosPlot(args, testing_records[0].seq, training_records[0].seq, testing_fasta, training_fasta, fn_alignments_pos_test, tp_alignments_pos_test, fn_genes_of_interest, 
			os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_FN_circos.png'), os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_FN_genes_circos.tsv'), shannon_scores, genomic_islands=gis_align)
	else:
		FNCircosPlot(args, testing_records[0].seq, training_records[0].seq, testing_fasta, training_fasta, fn_alignments_pos_test, tp_alignments_pos_test, fn_genes_of_interest, 
			os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_FN_circos.png'), os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_FN_genes_circos.tsv'), shannon_scores)
	
	# # # do FP analysis
	# # # blast FP reads to ncbi nt database
	# # with open(os.path.join(args.output_dir, f'{args.label}_FP_reads.fna'), "w") as outf:
	# # 	for k, v in test_readid_to_read.items():
	# # 		if k in fp_sequences:
	# # 			outf.write(f'>{k}\n{v}\n')
	# # RunBlast(args, os.path.join(args.output_dir, 'mapping'), os.path.join(args.output_dir, f'{args.label}_FP_reads.fna'), db=True)		

	# # # blast FP reads to train genome of label 1
	# # # create fasta file with all FP reads
	# # with open(os.path.join(args.output_dir, f'{args.label}_FP_reads.fna'), 'w') as outf:
	# # 	for k, v in readid_to_read.items():
	# # 		if k in fp_sequences:
	# # 			outf.write(f'>{k}\n{v}\n')

	# # RunBlast(args, os.path.join(args.output_dir, 'mapping'), os.path.join(args.output_dir, f'{args.label}_FP_reads.fna'), subject=[args.train_genomes_info[args.label][1]], outfilename=f'{args.output_dir}/mapping/FP_pos_train_blastn.out')

	# fp_labels = set([s.split('|')[1] for s in list(fp_sequences)])
	# fp_taxa = defaultdict(int)
	# print(f'# labels: {len(fp_labels)}')
	# outf = open(os.path.join(args.output_dir, 'FP_analysis', f'{args.label}_{args.prob_threshold}_FP_neg_genes.tsv'), 'w')

	# for label in fp_labels:
	# 	print(f'label: {label}')
	# 	label_testing_fasta = args.test_genomes_info[label][1]
	# 	label_testing_genome = args.test_genomes_info[label][0]

	# 	# get fp sequences of label and create fasta file
	# 	label_sequences = set([seq_id for seq_id in fp_sequences if seq_id.split('|')[1] == label])
		
	# 	mapping_output_dir = f'{args.output_dir}/blast/label0/testing-genome/{label}'
	# 	if not os.path.isdir(mapping_output_dir):
	# 		os.makedirs(mapping_output_dir)

	# 	with open(os.path.join(mapping_output_dir, f'{label}_FP_reads.fna'), 'w') as outf:
	# 		for k, v in test_readid_to_read.items():
	# 			if k in label_sequences:
	# 				outf.write(f'>{k}\n{v}\n')

	# 	# run blast
	# 	RunBlast(args, mapping_output_dir, os.path.join(mapping_output_dir, f'{label}_FP_reads.fna'), subject=[label_testing_fasta], outfilename=os.path.join(mapping_output_dir, 'test_test_blastn.out'))

	# 	# get alignments info
	# 	fp_alignments = GetReadsAlignments(label_sequences, os.path.join(mapping_output_dir, 'test_test_blastn.out'), test_sequence_length, seq_to_labels, os.path.join(args.output_dir, f'FP_neg_test_neg_test_{args.prob_threshold}_mapping_info.tsv'))
	# 	if label_testing_genome == 'GCF_004421065.1':
	# 		print(fp_alignments)
	# 	# get annotations info
	# 	neg_test_annot_info, _ = GetAnnotInfo(args, label_testing_genome, input_dir)

	# 	if len(neg_test_annot_info) != 0:
	# 		# get genes 
	# 		_ = GetGenes(args, label, os.path.join(args.output_dir, 'FP_analysis'), neg_test_annot_info, fp_alignments, test_sequence_length, test_readid_to_read, 'FP')

	# 	else:
	# 		print(f'No annotations for genome {label_testing_genome}')

	# 	# monitor number of sequences per label
	# 	fp_taxa[label] = len(label_sequences)

	# genetypes_files = glob.glob(os.path.join(args.output_dir, 'FP_analysis', f'*_FP_genes_type_{args.prob_threshold}.tsv'))
	# functions_files = glob.glob(os.path.join(args.output_dir, 'FP_analysis', f'*_FP_functions_{args.prob_threshold}.tsv'))
	# geneinfo_files = glob.glob(os.path.join(args.output_dir, 'FP_analysis', f'*_FP_genes_info_{args.prob_threshold}.tsv'))
	# readswogenesinfo_files = glob.glob(os.path.join(args.output_dir, 'FP_analysis', f'*_FP_reads_wo_gene_{args.prob_threshold}.tsv'))
	# readswogenesfna_files = glob.glob(os.path.join(args.output_dir, 'FP_analysis', f'*_FP_reads_wo_gene_{args.prob_threshold}.fna'))

	# ConcatenateFiles(genetypes_files, os.path.join(args.output_dir, f'{args.label}_FP_genes_type_{args.prob_threshold}.tsv'), "gene_type")
	# ConcatenateFiles(functions_files, os.path.join(args.output_dir, f'{args.label}_FP_functions_{args.prob_threshold}.tsv'), "function")
	# ConcatenateFiles(geneinfo_files, os.path.join(args.output_dir, f'{args.label}_FP_genes_info_{args.prob_threshold}.tsv'), "gene_info")
	# ConcatenateFiles(readswogenesinfo_files, os.path.join(args.output_dir, f'{args.label}_FP_reads_wo_gene_{args.prob_threshold}.tsv'), "reads_wo_genes")
	# ConcatenateFiles(readswogenesfna_files, os.path.join(args.output_dir, f'{args.label}_FP_reads_wo_gene_{args.prob_threshold}.fna'), "reads_wo_genes")

	# fp_taxa_sorted = dict(sorted(fp_taxa.items(), key=lambda item: item[1], reverse=True))
	# with open(os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_FP_neg_taxa.tsv'), 'w') as f:
	# 	for k, v in fp_taxa_sorted.items():
	# 		f.write(f'{k}\t{args.dl_toda_tax[k]}\t{v}\n')
	
	# with open(os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_FP_reads.fq'), 'w') as f:
	# 	f.write(''.join([f'>{r}\n{test_readid_to_read[r]}\n' for r in list(fp_sequences)]))




