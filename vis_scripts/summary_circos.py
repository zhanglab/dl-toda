import sys
import os
import glob
import argparse
import math
import statistics
from collections import defaultdict
from pycirclize import Circos
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from dataprep_scripts.utils import load_fq_file
from vis_scripts.parse_samfile import LoadData, GetCoverageOfSample

def GetTaxaAndMappingInfo(alignments, label, sequence_length, genome_size, type):
	""" return list with number of unique taxon per position on the target genome"""
	mapped_taxa_info = defaultdict(list)
	mapped_pos_info = [0 for i in range(genome_size)]

	if type == 'FN':
		for read_id in alignments.keys():
			mapped_taxa = list(set(alignments[read_id].values()))
			if label in mapped_taxa:
				mapped_taxa.remove(label)
			start_pos = int(line.rstrip().split('\t')[3])
			for i in range(start_pos, start_pos+sequence_length[read_id]+1, 1):
				mapped_taxa_info[i] += mapped_taxa
				mapped_pos_info[i-1] += 1
	
	elif type == 'FP':
		for read_id in alignments.keys():
			read_label = read_id.split('|')[1]
			start_pos = int(line.rstrip().split('\t')[3])
			for i in range(start_pos, start_pos+sequence_length[read_id]+1, 1):
				mapped_taxa_info[i] += [read_label]
				mapped_pos_info[i-1] += 1

	elif type == 'TP':
		for read_id in alignments.keys():
			start_pos = int(line.rstrip().split('\t')[3])
			for i in range(start_pos, start_pos+sequence_length[read_id]+1, 1):
				mapped_pos_info[i-1] += 1

	taxa_count = [0 for i in range(genome_size)]
	for k, v in mapped_taxa_info.items():
		taxa_count[k-1] = len(set(v))

	return taxa_count, mapped_pos_info


def GetAlignmentsInfo(sequences, samfile):
	alignments = defaultdict(dict)
	mapped_reads_id = []
	unmapped_reads_id = []
	with open(samfile, 'r') as f:
		for line in f:
			if line.rstrip().split('\t')[0][:3] not in ['@PG', '@SQ', '@HD']:
				read_id = line.rstrip().split('\t')[0]
				if read_id in sequences:
					if line.rstrip().split('\t')[5] != '*':
						seq_id = line.rstrip().split('\t')[2]
						seq_label = seq_to_labels[seq_id]
						start_pos = int(line.rstrip().split('\t')[3])
						mapping_score =int(line.rstrip().split('\t')[3])
						alignments[read_id][seq_label] = [start_pos, mapping_score]
						mapped_reads_id.append(read_id)
					else:
						unmapped_reads_id.append(read_id)

	return alignments, mapped_reads_id, unmapped_reads_id


def GetSeqLength(sequences_id, sequence_length, type):
	seq_length_info = [sequence_length[s] for s in sequences_id]
	print(f'{type}\tmean: {statistics.mean(seq_length_info)}\tmedian: {statistics.median(seq_length_info)}\tmax: {max(seq_length_info)}\tmin: {min(seq_length_info)}')


def PlotCircos(genome_pos, train_pos_coverage, fn_mapped_pos_info, tp_mapped_pos_info, fp_mapped_pos_info, fn_taxa_count, fp_taxa_count, output_filename):
	# initialize a single circos sector
	sectors = {'genome': len(genome_pos)}
	circos = Circos(sectors=sectors, space=14)
	for sector in circos.sectors:
		# add track for positions of the genome
		genome_track = sector.add_track((98, 100))
		genome_track.axis(fc="lightgrey")
		interval = str(int(len(genome_pos)/5))
		interval = int(interval[0]+ '0'*(len(interval)-1))
		genome_x = list(range(0,len(genome_pos),interval))
		base_pos_ticks = [genome_pos[i] for i in genome_x]
		genome_x_labels = ['1 bp'] + [f'{i/1000} Kb' for i in base_pos_ticks[1:]]
		genome_track.xticks(genome_x, genome_x_labels)
		genome_track.xticks_by_interval(100000, tick_length=1, show_label=False)
		print(f'added genome track')
		# add track for coverage of the genome
		cov_track = sector.add_track((92, 97))
		cov_track.axis()
		cov_y = list(range(min([int(i) for i in train_pos_coverage]), max([math.ceil(j) for j in train_pos_coverage])+1, 4))
		cov_y_labels = list(map(str, cov_y))
		cov_track.yticks(cov_y, cov_y_labels)
		cov_track.line(list(range(0,len(genome_pos),1)), train_pos_coverage, color="#00A5E3")
		print(f'added coverage track')
		# add track for FN positions
		fn_count_track = sector.add_track((86, 91))
		fn_count_track.axis()
		fn_count_y = list(range(min(fn_mapped_pos_info), max(fn_mapped_pos_info)+1, 10))
		print(f'min(test_pos_count): {min(fn_mapped_pos_info)}\tmax(test_pos_count)+1: {max(fn_mapped_pos_info)+1}')
		fn_count_y_labels = list(map(str, test_count_y))
		fn_count_track.yticks(test_count_y, test_count_y_labels)
		fn_count_x = genome_pos
		fn_count_x_values = fn_mapped_pos_info
		fn_count_track.heatmap(fn_mapped_pos_info, cmap="viridis")
		# test_count_track.line(test_count_x, test_count_x_values, color="#9e1369")
		# save figure
		circos.savefig(output_filename)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--pos_test_neg_train', type=str, help='path to sam file with mapping of testing sequences from label of interest to training genomes from other labels')
	parser.add_argument('--pos_test_pos_train', type=str, help='path to sam file with mapping of testing sequences from label of interest to training genome from label of interest')
	parser.add_argument('--neg_test_pos_train', type=str, help='path to sam file with mapping of testing sequences from other labels to training genome from label of interest')
	parser.add_argument('--pos_train_pos_train', type=str, help='path to sam file with mapping of training sequences from label of interest to training genome from label of interest')
	parser.add_argument('--testing_fq_file', type=str, help='path to fastq file containing all testing reads (+ and - class)')
	parser.add_argument('--label', type=str, help='label of species investigated')
	parser.add_argument('--sequences_info', type=str, help='path to file mapping labels of species in model to training sequence ids')
	parser.add_argument('--prob_threshold', type=float, help='probability score threshold')
	parser.add_argument('--test_genome_size', type=int, help='size in bp of genome of interest')
	parser.add_argument('--rank', type=str, help='taxonomic rank investigated', choices=['species','genus','family','order','class', 'phylum'])
	parser.add_argument('--testing_results', type=str, help='path to file containing testing results')
	parser.add_argument('--dltoda_taxonomy', type=str, help='path to file containing dltoda taxonomy')
	parser.add_argument('--output_dir', type=str, help='path to output directory', default=os.getcwd())
	args = parser.parse_args()
	
	sequences = load_fq_file(args.testing_fq_file, 4)
	sequence_length = {line.split('\n')[0][1:]: len(line.split('\n')[1]) for line in sequences}
	print(sequence_length['seq|100|num_0'])

	# get FN and FP sequences
	fn_sequences = set()
	fp_sequences = set()
	tp_sequences = set()
	with open(args.testing_results, 'r') as f:
		for count, line in enumerate(f):
			prob = float(line.rstrip().split('\t')[2])
			if prob >= args.prob_threshold:
				if line.rstrip().split('\t')[0] == '1' and line.rstrip().split('\t')[1] == '0':
					fn_sequences.add(sequences[count].split('\n')[0][1:])
				if line.rstrip().split('\t')[0] == '0' and line.rstrip().split('\t')[1] == '1':
					fp_sequences.add(sequences[count].split('\n')[0][1:])
				if line.rstrip().split('\t')[0] == '1' and line.rstrip().split('\t')[1] == '1':
					tp_sequences.add(sequences[count].split('\n')[0][1:])

	print(f'#FN for label {args.label}: {len(fn_sequences)}')
	print(f'#FP for label {args.label}: {len(fp_sequences)}')
	print(f'#TP for label {args.label}: {len(tp_sequences)}')
	GetSeqLength(list(fn_sequences), sequence_length, 'label 239 testing FN sequences')
	GetSeqLength(list(fp_sequences), sequence_length, 'label 239 testing FP sequences')
	GetSeqLength(list(tp_sequences), sequence_length, 'label 239 testing TP sequences')

	# get association between sequences and labels
	with open(args.sequences_info, 'r') as f:
		content = f.readlines()
		seq_to_labels = {line.rstrip().split('\t')[1]: line.rstrip().split('\t')[0] for line in content}


	# get alignments info for FN, FP and TP reads
	fn_alignments, fn_mapped_reads_id, fn_unmapped_reads_id = GetAlignmentsInfo(fn_sequences, args.pos_test_neg_train)
	fp_alignments, fp_mapped_reads_id, fp_unmapped_reads_id = GetAlignmentsInfo(fp_sequences, args.neg_test_pos_train)
	tp_alignments, tp_mapped_reads_id, tp_unmapped_reads_id = GetAlignmentsInfo(tp_sequences, args.pos_test_neg_train)

	GetSeqLength(fn_mapped_reads_id, sequence_length, 'label 239 mapped testing FN sequences to label 239 training genome')
	GetSeqLength(fp_mapped_reads_id, sequence_length, 'other labels mapped testing FP sequences to label 239 training genome')
	GetSeqLength(tp_mapped_reads_id, sequence_length, 'label 239 mapped testing TP sequences to label 239 training genome')
	GetSeqLength(fn_unmapped_reads_id, sequence_length, 'label 239 unmapped testing FN sequences to label 239 training genome')
	GetSeqLength(fp_unmapped_reads_id, sequence_length, 'other labels unmapped testing FP sequences to label 239 training genome')
	GetSeqLength(tp_unmapped_reads_id, sequence_length, 'label 239 unmapped testing TP sequences to label 239 training genome')

	# get coverage of training genome with training sequences
	ref_info, alignments = LoadData(args.pos_train_pos_train)
	training_genome_size = ref_info[0][1]
	print(f'training_genome_size: {training_genome_size}\ttest_genome_size: {args.test_genome_size}')
	dict_coverage, reads_info = GetCoverageOfSample(alignments[ref_info[0][0]], training_genome_size, label=args.label)
	print(f'# testing reads ')
	train_pos_coverage = [dict_coverage[i] for i in range(training_genome_size)]
	with open(os.path.join(args.output_dir, f'{args.label}_'), 'w') as f:
		for i in range(len(train_pos_coverage)):
			f.write(f'{i+1}\t{train_pos_coverage[i]}')
	print(f'mean: {statistics.mean(train_pos_coverage)}\tmedian: {statistics.median(train_pos_coverage)}\tmin: {min(train_pos_coverage)}\tmax: {max(train_pos_coverage)}')

	# get number of unique taxa mapped by FN testing reads per position of the label's training genome
	fn_taxa_count, fn_mapped_pos_info = GetTaxaAndMappingInfo(fn_alignments, args.label, sequence_length, training_genome_size, 'FN')
	# get number of unique taxa mapped by FP testing reads per position of the label's training genome
	fp_taxa_count, fp_mapped_pos_info = GetTaxaAndMappingInfo(fp_alignments, args.label, sequence_length, training_genome_size, 'FP')
	# get positions on the label's training genome where TP testing reads map
	_, tp_mapped_pos_info = GetTaxaAndMappingInfo(fp_alignments, args.label, sequence_length, training_genome_size, 'FP')
	
	train_genome_pos = list(range(1, training_genome_size+1, 1))
	# create circos plot with FN reads info
	PlotCircos(train_genome_pos, train_pos_coverage, fn_mapped_pos_info, tp_mapped_pos_info, fp_mapped_pos_info, fn_taxa_count, fp_taxa_count, os.path.join(args.output_dir, f'{args.label}_false_negatives.png'))

