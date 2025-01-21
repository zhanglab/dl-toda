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

def GetSeqLength(samfile, target_sequences, alignments):
	# get average FN sequence length and number of mapped taxa at each mapped position of the testing genome
	seq_length_info = defaultdict(list)
	mapped_taxa_info = defaultdict(list)
	with open(samfile, 'r') as f:
		for line in f:
			if line.rstrip().split('\t')[0][:3] not in ['@PG', '@SQ', '@HD'] and line.rstrip().split('\t')[5] != '*':
				seq_id = line.rstrip().split('\t')[0]
				if seq_id in target_sequences:
					mapped_taxa = list(alignments[seq_id].values())
					if args.label in mapped_taxa:
						mapped_taxa.remove(args.label)

					start_pos = int(line.rstrip().split('\t')[3])
					for i in range(start_pos, start_pos+sequence_length[seq_id]+1, 1):
						seq_length_info[i].append(sequence_length[seq_id])
						mapped_taxa_info[i] += mapped_taxa

	return seq_length_info, mapped_taxa_info


def PlotCirclesFnTrainGenome(genome_positions, train_pos_coverage, test_pos_count, output_dir, label):
	# initialize a single circos sector
	sectors = {'genome': len(genome_positions)}
	circos = Circos(sectors=sectors, space=14)

	for sector in circos.sectors:
		# add track for positions of the label's testing genome
		genome_track = sector.add_track((98, 100))
		genome_track.axis(fc="lightgrey")
		interval = str(int(len(genome_positions)/5))
		interval = int(interval[0]+ '0'*(len(interval)-1))
		genome_x = list(range(0,len(genome_positions),interval))
		base_pos_ticks = [genome_positions[i] for i in genome_x]
		genome_x_labels = ['1 bp'] + [f'{i/1000} Kb' for i in base_pos_ticks[1:]]
		print(genome_x_labels)
		print(genome_x)
		genome_track.xticks(genome_x, genome_x_labels)
		genome_track.xticks_by_interval(100000, tick_length=1, show_label=False)
		print(f'added genome track')
		# add track for coverage of label's training genome
		cov_track = sector.add_track((85, 95))
		cov_track.axis()
		cov_y = list(range(min([int(i) for i in train_pos_coverage]), max([math.ceil(j) for j in train_pos_coverage])+1, 4))
		cov_y_labels = list(map(str, cov_y))
		cov_track.yticks(cov_y, cov_y_labels)
		cov_track.line(list(range(0,len(genome_positions),1)), train_pos_coverage, color="#00A5E3")
		print(f'added coverage track')
		# add track for count of unique taxa per position
		test_count_track = sector.add_track((72, 82))
		test_count_track.axis()
		test_count_y = list(range(min(test_pos_count), max(test_pos_count)+1, 10))
		test_count_y_labels = list(map(str, test_count_y))
		test_count_track.yticks(test_count_y, test_pos_count)
		test_count_x = genome_positions
		test_count_x_values = test_pos_count
		test_count_track.line(test_count_x, test_count_x_values, color="#9e1369")
		# save figure
		circos.savefig(os.path.join(output_dir, f'circos_fn_train_genome{label}.png'))


def PlotCirclesFnTestGenome(genome_positions, test_genome_seq_length, test_genome_taxa_count, output_dir, label):
	# initialize a single circos sector
	sectors = {'genome': len(genome_positions)}
	circos = Circos(sectors=sectors, space=14)

	for sector in circos.sectors:
		# add track for positions of the label's testing genome
		genome_track = sector.add_track((98, 100))
		genome_track.axis(fc="lightgrey")
		interval = str(int(len(genome_positions)/5))
		interval = int(interval[0]+ '0'*(len(interval)-1))
		genome_x = list(range(0,len(genome_positions),interval))
		base_pos_ticks = [genome_positions[i] for i in genome_x]
		genome_x_labels = ['1 bp'] + [f'{i/1000} Kb' for i in base_pos_ticks[1:]]
		print(genome_x_labels)
		print(genome_x)
		genome_track.xticks(genome_x, genome_x_labels)
		genome_track.xticks_by_interval(100000, tick_length=1, show_label=False)
		print(f'added genome track')
		# add track for average sequence length of FN testing sequences
		seq_length_track = sector.add_track((85, 95))
		seq_length_track.axis()
		seq_length_y = list(range(min([int(i) for i in test_genome_seq_length]), max([math.ceil(j) for j in test_genome_seq_length])+1, 4))
		seq_length_y_labels = list(map(str, seq_length_y))
		seq_length_track.yticks(seq_length_y, seq_length_y_labels)
		seq_length_track.line(list(range(0,len(genome_positions),1)), test_genome_seq_length, color="#00A5E3")
		print(f'added sequence length track')
		# add track for count of unique taxa per position
		taxa_count_track = sector.add_track((72, 82))
		taxa_count_track.axis()
		taxa_count_y = list(range(min(test_genome_taxa_count), max(test_genome_taxa_count)+1, 10))
		taxa_count_y_labels = list(map(str, taxa_count_y))
		taxa_count_track.yticks(taxa_count_y, taxa_count_y_labels)
		taxa_count_x = genome_positions
		taxa_count_x_values = test_genome_taxa_count
		taxa_count_track.line(taxa_count_x, taxa_count_x_values, color="#9e1369")
		# save figure
		circos.savefig(os.path.join(output_dir, f'circos_fn_test_genome{label}.png'))

# def PlotCirclesTrainGenome(genome_positions, unique_taxa_count, total_taxa_count, confidence_scores, pos_coverage, output_dir, label):
	
# 	# initialize a single circos sector
# 	sectors = {'genome': len(genome_positions)}
# 	circos = Circos(sectors=sectors, space=14)

# 	for sector in circos.sectors:
# 		# add track for positions of the label's testing genome
# 		genome_track = sector.add_track((98, 100))
# 		genome_track.axis(fc="lightgrey")
# 		interval = str(int(len(genome_positions)/5))
# 		interval = int(interval[0]+ '0'*(len(interval)-1))
# 		genome_x = list(range(0,len(genome_positions),interval))
# 		base_pos_ticks = [genome_positions[i] for i in genome_x]
# 		genome_x_labels = ['1 bp'] + [f'{i/1000} Kb' for i in base_pos_ticks[1:]]
# 		print(genome_x_labels)
# 		print(genome_x)
# 		genome_track.xticks(genome_x, genome_x_labels)
# 		genome_track.xticks_by_interval(100000, tick_length=1, show_label=False)
# 		print(f'added genome track')
# 		# add track for coverage of label's training genome
# 		cov_track = sector.add_track((85, 95))
# 		cov_track.axis()
# 		cov_y = list(range(min([int(i) for i in pos_coverage]), max([math.ceil(j) for j in pos_coverage])+1, 4))
# 		cov_y_labels = list(map(str, cov_y))
# 		cov_track.yticks(cov_y, cov_y_labels)
# 		cov_track.line(list(range(0,len(genome_positions),1)), pos_coverage, color="#00A5E3")
# 		print(f'added coverage track')
# 		# add track for count of unique taxa per position
# 		unique_taxa_track = sector.add_track((72, 82))
# 		unique_taxa_track.axis()
# 		unique_taxa_y = list(range(min(unique_taxa_count), max(unique_taxa_count)+1, 10))
# 		unique_taxa_y_labels = list(map(str, unique_taxa_y))
# 		unique_taxa_track.yticks(unique_taxa_y, unique_taxa_y_labels)
# 		unique_taxa_x = genome_positions
# 		unique_taxa_x_values = unique_taxa_count
# 		unique_taxa_track.line(unique_taxa_x, unique_taxa_x_values, color="#9e1369")
# 		# # add track for count of total taxa per position
# 		# total_taxa_track = sector.add_track((59, 69))
# 		# total_taxa_track.axis()
# 		# total_taxa_y = list(range(int(min(total_taxa_count)), int(max(total_taxa_count))+1, 1000))
# 		# print(total_taxa_y)
# 		# total_taxa_y_labels = list(map(str, total_taxa_y))
# 		# print(total_taxa_y_labels)
# 		# total_taxa_track.yticks(total_taxa_y, total_taxa_y_labels)
# 		# total_taxa_x = genome_positions
# 		# total_taxa_x_values = total_taxa_count
# 		# total_taxa_track.line(total_taxa_x, total_taxa_x_values, color="#465d66")
# 		# # add track for confidence scores obtained of the label's testing reads
# 		# pos_cs_track = sector.add_track((46, 56))
# 		# pos_cs_track.axis()
# 		# pos_cs_y = [0.0, 0.5, 1.0]
# 		# pos_cs_y_labels = list(map(str, pos_cs_y))
# 		# pos_cs_track.yticks(pos_cs_y, pos_cs_y_labels)
# 		# pos_cs_x = genome_positions
# 		# pos_cs_x_values = confidence_scores
# 		# pos_cs_track.scatter(pos_cs_x, pos_cs_x_values, color="#FC6238")
# 		# print(f'added pos cs track')
# 		# save figure
# 		circos.savefig(os.path.join(output_dir, f'circos_{label}.png'))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--test_train_samfiles', type=str, help='path to directory containing sam files with mapping of testing sequences to training genomes')
	parser.add_argument('--train_train_samfile', type=str, help='path to sam file with mapping of training sequences to training genome')
	parser.add_argument('--test_test_samfile', type=str, help='path to sam file with mapping of testing sequences to testing genome')
	parser.add_argument('--testing_fq_file', type=str, help='path to fastq file containing all testing reads (+ and - class)')
	parser.add_argument('--label', type=str, help='label of species investigated')
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
	with open(args.testing_results, 'r') as f:
		for count, line in enumerate(f):
			prob = float(line.rstrip().split('\t')[2])
			if prob >= args.prob_threshold:
				if line.rstrip().split('\t')[0] == '1' and line.rstrip().split('\t')[1] == '0':
					fn_sequences.add(sequences[count].split('\n')[0][1:])
				if line.rstrip().split('\t')[0] == '0' and line.rstrip().split('\t')[1] == '1':
					fp_sequences.add(sequences[count].split('\n')[0][1:])

	print(f'#FN for label {args.label}: {len(fn_sequences)}')
	print(f'#FP for label {args.label}: {len(fp_sequences)}')

	# get alignment info for FN and FP sequences with training genomes
	fn_alignments = defaultdict(dict)
	fp_alignments = defaultdict(dict)
	samfiles = glob.glob(os.path.join(args.test_train_samfiles, '*.sam'))
	fn_mapped_labels = defaultdict(int)
	for s in samfiles:
		with open(s, 'r') as f:
			sam_label = s.split('/')[-1].split('_')[0]
			for line in f:
				if line.rstrip().split('\t')[0][:3] not in ['@PG', '@SQ', '@HD'] and line.rstrip().split('\t')[5] != '*':
					seq_id = line.rstrip().split('\t')[0]
					if seq_id in fn_sequences:
						start_pos = int(line.rstrip().split('\t')[3])
						fn_alignments[seq_id][sam_label] = start_pos
						fn_mapped_labels[sam_label] += 1
					if seq_id in fp_sequences:
						start_pos = int(line.rstrip().split('\t')[3])
						fp_alignments[seq_id][sam_label] = start_pos

	# get coverage of training genome with training sequences
	ref_info, alignments = LoadData(args.train_train_samfile)
	training_genome_size = ref_info[0][1]
	dict_coverage, reads_info = GetCoverageOfSample(alignments[ref_info[0][0]], training_genome_size)
	train_pos_coverage = [dict_coverage[i] for i in range(training_genome_size)]

	# get positions of training genome mapped by FN testing sequences
	train_genome_fn_count = [0 for i in range(training_genome_size)]
	mapped_fn_seq_count = 0
	unmapped_fn_seq_length = []
	for seq_id in list(fn_sequences):
		if seq_id in fn_alignments and args.label in fn_alignments[seq_id]:
			mapped_fn_seq_count += 1
			start_pos = fn_alignments[seq_id][args.label]
			for i in range(start_pos, start_pos+sequence_length[seq_id]+1, 1):
				train_genome_fn_count[i-1] += 1
		else:
			unmapped_fn_seq_length.append(sequence_length[seq_id])
	print(f'train_genome_fn_count: {len(train_genome_fn_count)}\t{train_genome_fn_count[:10]}')
	print(f'# FN sequences mapped to training genome: {mapped_fn_seq_count}')
	print(f'# FN sequences unmapped to training genome: {len(unmapped_fn_seq_length)}\t{statistics.mean(unmapped_fn_seq_length)}\t{statistics.median(unmapped_fn_seq_length)}\t{min(unmapped_fn_seq_length)}\t{max(unmapped_fn_seq_length)}')

	# # get positions of training genome mapped by FP testing sequences
	# train_genome_fp_count = [0 for i in range(training_genome_size)]  # key = training genome position, value = number of mapped FN sequences
	# mapped_fp_seq_count = 0
	# unmapped_fp_seq_length = []
	# for seq_id in list(fp_sequences):
	# 	if seq_id in fp_alignments and args.label in fp_alignments[seq_id]:
	# 		mapped_fp_seq_count += 1
	# 		start_pos = fp_alignments[seq_id][args.label]
	# 		for i in range(start_pos, start_pos+sequence_length[seq_id]+1, 1):
	# 			train_genome_fp_count[i-1] += 1
	# 	else:
	# 		unmapped_fp_seq_length.append(sequence_length[seq_id])
	# print(f'# FP sequences mapped to training genome: {mapped_fp_seq_count}')
	# print(f'# FP sequences unmapped to training genome: {len(unmapped_fp_seq_length)}\t{statistics.mean(unmapped_fp_seq_length)}\t{statistics.mean(unmapped_fp_seq_length)}\t{min(unmapped_fp_seq_length)}\t{max(unmapped_fp_seq_length)}')

	# get average FP sequence length and number of taxa that were misclassified at each mapped position of the training genome	
	fn_seq_length_info, fn_mapped_taxa_info = GetSeqLength(args.test_test_samfile, list(fn_sequences), fn_alignments)

	fn_test_genome_seq_length = [0 for i in range(args.test_genome_size)]
	print(f'fn_test_genome_seq_length: {fn_test_genome_seq_length[:10]}\t{len(fn_test_genome_seq_length)}')
	for k, v in fn_seq_length_info.items():
		fn_test_genome_seq_length[k-1] = statistics.mean(v)

	fn_test_genome_taxa_count = [0 for i in range(args.test_genome_size)]
	for k, v in fn_mapped_taxa_info.items():
		fn_test_genome_taxa_count[k-1] = len(set(v))

	train_genome_pos = list(range(1, training_genome_size+1, 1))
	PlotCirclesFnTrainGenome(train_genome_pos, train_pos_coverage, train_genome_fn_count, args.output_dir, args.label)

	test_genome_pos = list(range(1, args.test_genome_size+1, 1))
	PlotCirclesFnTestGenome(test_genome_pos, fn_test_genome_seq_length, fn_test_genome_taxa_count, args.output_dir, args.label)

	

	# print(fn_alignments)
	# print(len(mapped_labels))
	# mapped_reads_count = list(mapped_labels.values())
	# max_value = max(mapped_reads_count)
	# print(max_value)
	# print(statistics.mean(mapped_reads_count), statistics.median(mapped_reads_count), min(mapped_reads_count))
	# for k, v in mapped_labels.items():
	# 	if v == max_value:
	# 		print(k, max_value)











# import sys
# import os
# import glob
# import json
# from pycirclize import Circos
# sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
# from dataprep_scripts.utils import load_fq_file
# from vis_scripts.parse_samfile import LoadData, GetCoverage
# from collections import defaultdict
# import random
# import numpy as np
# import math
# import pandas as pd
# import multiprocessing as mp
# import argparse
# import statistics
# from matplotlib import colormaps
# import matplotlib.colors as mcolors
# import matplotlib.pyplot as plt
# from collections import Counter

# def PlotCircles(genome_positions, unique_taxa_count, total_taxa_count, confidence_scores, pos_coverage, output_dir, label):
	
# 	# initialize a single circos sector
# 	sectors = {'genome': len(genome_positions)}
# 	circos = Circos(sectors=sectors, space=14)

# 	for sector in circos.sectors:
# 		# add track for positions of the label's testing genome
# 		genome_track = sector.add_track((98, 100))
# 		genome_track.axis(fc="lightgrey")
# 		interval = str(int(len(genome_positions)/5))
# 		interval = int(interval[0]+ '0'*(len(interval)-1))
# 		genome_x = list(range(0,len(genome_positions),interval))
# 		base_pos_ticks = [genome_positions[i] for i in genome_x]
# 		genome_x_labels = ['1 bp'] + [f'{i/1000} Kb' for i in base_pos_ticks[1:]]
# 		print(genome_x_labels)
# 		print(genome_x)
# 		genome_track.xticks(genome_x, genome_x_labels)
# 		genome_track.xticks_by_interval(100000, tick_length=1, show_label=False)
# 		print(f'added genome track')
# 		# add track for coverage of label's training genome
# 		cov_track = sector.add_track((85, 95))
# 		cov_track.axis()
# 		cov_y = list(range(min([int(i) for i in pos_coverage]), max([math.ceil(j) for j in pos_coverage])+1, 4))
# 		cov_y_labels = list(map(str, cov_y))
# 		cov_track.yticks(cov_y, cov_y_labels)
# 		cov_track.line(list(range(0,len(genome_positions),1)), pos_coverage, color="#00A5E3")
# 		print(f'added coverage track')
# 		# add track for count of unique taxa per position
# 		unique_taxa_track = sector.add_track((72, 82))
# 		unique_taxa_track.axis()
# 		unique_taxa_y = list(range(min(unique_taxa_count), max(unique_taxa_count)+1, 10))
# 		unique_taxa_y_labels = list(map(str, unique_taxa_y))
# 		unique_taxa_track.yticks(unique_taxa_y, unique_taxa_y_labels)
# 		unique_taxa_x = genome_positions
# 		unique_taxa_x_values = unique_taxa_count
# 		unique_taxa_track.line(unique_taxa_x, unique_taxa_x_values, color="#9e1369")
# 		# add track for count of total taxa per position
# 		total_taxa_track = sector.add_track((59, 69))
# 		total_taxa_track.axis()
# 		total_taxa_y = list(range(int(min(total_taxa_count)), int(max(total_taxa_count))+1, 1000))
# 		print(total_taxa_y)
# 		total_taxa_y_labels = list(map(str, total_taxa_y))
# 		print(total_taxa_y_labels)
# 		total_taxa_track.yticks(total_taxa_y, total_taxa_y_labels)
# 		total_taxa_x = genome_positions
# 		total_taxa_x_values = total_taxa_count
# 		total_taxa_track.line(total_taxa_x, total_taxa_x_values, color="#465d66")
# 		# add track for confidence scores obtained of the label's testing reads
# 		pos_cs_track = sector.add_track((46, 56))
# 		pos_cs_track.axis()
# 		pos_cs_y = [0.0, 0.5, 1.0]
# 		pos_cs_y_labels = list(map(str, pos_cs_y))
# 		pos_cs_track.yticks(pos_cs_y, pos_cs_y_labels)
# 		pos_cs_x = genome_positions
# 		pos_cs_x_values = confidence_scores
# 		pos_cs_track.scatter(pos_cs_x, pos_cs_x_values, color="#FC6238")
# 		print(f'added pos cs track')
# 		# save figure
# 		circos.savefig(os.path.join(output_dir, f'circos_{label}.png'))


# def GetTaxa(args, dltoda_tax, genome_positions, mapping_info):
# 	# load data about alignments of testing reads to training genomes
# 	samfiles = glob.glob(os.path.join(args.train_samfiles, '*.sam'))
# 	samfiles.remove(os.path.join(args.train_samfiles, f'{args.label}_results.sam'))
# 	print(f'# sam files: {len(samfiles)}')
# 	mapped_species = defaultdict(set) # key = position in testing genome, value = list of taxa with a training genome to which the testing read was mapped to
# 	unique_mapped_taxa = set()
	
# 	for sam in samfiles:
# 		# get species label of training genome from SAM filename
# 		sam_label = sam.rstrip().split('/')[-1].split('_')[0]
		
# 		# get info about alignment
# 		_, sam_alignment = LoadData(sam)
		
# 		for sam_ref, sam_info in sam_alignment.items():
# 			for read_info in sam_info:
# 				read_id = read_info[0]
# 				# get start and end positions of alignment on the testing genome
# 				if read_id in mapping_info:
# 					start_position = mapping_info[read_id][0]
# 					end_position = mapping_info[read_id][1]
# 					if read_id == 'seq|492|num_16073':
# 						print(read_id, start_position, end_position, end_position-start_position)
# 					for p in range(start_position, end_position, 1):
# 						mapped_species[p].add(sam_label)
# 						unique_mapped_taxa.add(dltoda_tax[sam_label])

#     # create dataframe with rows = positions in testing genome and columns = mapped taxa
# 	unique_mapped_taxa = list(unique_mapped_taxa)
# 	row_num = len(genome_positions)
# 	col_num = len(unique_mapped_taxa)
# 	print(f'Size of testing genome: {row_num}')
    
# 	matrix = np.zeros((row_num, col_num))
# 	row_names = [i for i in range(1,row_num+1,1)]
# 	col_names = [i for i in unique_mapped_taxa]
# 	print(col_names)
# 	print(len(row_names), row_names[0], row_names[-1])
# 	df = pd.DataFrame(matrix, index=row_names, columns=col_names)
# 	print(df)
# 	for pos, labels_list in mapped_species.items():
# 		for l in list(labels_list):
# 			# get taxon of label at given rank
# 			taxon = dltoda_tax[l]
# 			df.loc[pos, taxon] += 1
# 	print(df)
# 	df.to_csv(os.path.join(args.output_dir, f'taxa_read_count_{args.label}_df.csv'), index=False)
# 	return df


# def GetConfidenceScores(args, testing_genome_length, mapping_info, reads_id):
# 	# load testing results
# 	dict_confidence_scores = defaultdict(list)
    
# 	with open(args.testing_results, 'r') as f:
# 		content = f.readlines()
# 		testing_results_data = [line.rstrip().split('\t')[2] for line in content]
    
# 	assert len(testing_results_data) == len(reads_id), "the number of reads id does not match the number of reads tested"

# 	for i, r in enumerate(reads_id):
# 		label = r.split('|')[1]
#     	# only get confidence scores of reads belonging to label
# 		if label == args.label:
# 			if r in mapping_info:
# 				start_pos = mapping_info[r][0]
# 				end_pos = mapping_info[r][1]
# 				confidence_score = float(testing_results_data[i])
# 				for i in range(start_pos, end_pos, 1):
# 					dict_confidence_scores[i].append(confidence_score)

# 	confidence_scores = []	
# 	for i in range(1, testing_genome_length+1, 1):
# 		if i in dict_confidence_scores:
# 			confidence_scores.append(statistics.mean(dict_confidence_scores[i]))
# 		else:
# 			confidence_scores.append(0.0)

# 	return confidence_scores


# def GetInfoTestingGenome(args):

# 	# load data about alignments of testing reads to testing genome
# 	ref_info, alignments = LoadData(args.test_samfile)
	
# 	assert len(ref_info) == 1, f'{args.test_samfile} has more than 1 reference sequence'
	
# 	testing_genome_length = ref_info[0][1]
# 	testing_genome_pos = list(range(1, testing_genome_length+1, 1))
# 	dict_coverage, reads_info = GetCoverage(alignments[ref_info[0][0]], testing_genome_length)
# 	pos_coverage = [dict_coverage[i] for i in range(testing_genome_length)]

# 	# load fq file with testing reads --> required to identify reads that were not mapped to the reference testing genome (too short)
# 	reads = load_fq_file(args.fq_file, 4)
# 	reads_id = []
# 	dict_reads_length = {}
# 	for r in reads:
# 		read_id = r.split("\n")[0][1:]
# 		length = len(r.split("\n")[1])
# 		if r.split("\n")[0].split('|')[1] == args.label:
# 			dict_reads_length[read_id] = length
# 			reads_id.append(read_id)
# 	print(len(reads_id))

# 	# get reads that were not mapped to the testing genome and their length
# 	unmapped_reads = set(list(dict_reads_length.keys())).difference(set(list(reads_info.keys())))
# 	with open(os.path.join(args.output_dir, f'unmapped_reads_{args.label}.tsv'), 'w') as f:
# 		for r in unmapped_reads:
# 			read_label = r.split('|')[1]
# 			if read_label == args.label:
# 				f.write(f'{r}\t{dict_reads_length[r]}\n')

# 	return reads_id, pos_coverage, testing_genome_pos, reads_info


# def main():
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--train_samfiles', type=str, help='directory containing SAM files with alignment of testing reads to training genomes')
# 	parser.add_argument('--test_samfile', type=str, help='path to SAM file containing the alignment of testing reads to the testing genome')
# 	parser.add_argument('--fq_file', type=str, help='path to fastq file containing all testing reads (+ and - class)')
# 	parser.add_argument('--label', type=str, help='label of species investigated')
# 	parser.add_argument('--kmers', type=str, help='path to file containing of kmers of interest')
# 	parser.add_argument('--rank', type=str, help='taxonomic rank investigated', choices=['species','genus','family','order','class', 'phylum'])
# 	parser.add_argument('--testing_results', type=str, help='path to file containing testing results')
# 	parser.add_argument('--output_dir', type=str, help='path to output directory', default=os.getcwd())
# 	parser.add_argument('--input_dir', type=str, help='path to input directory', default=os.getcwd())
# 	parser.add_argument('--num_processes', type=int, default=8)
# 	args = parser.parse_args()

#     # define path to taxonomy of genomes in dltoda
# 	ranks_index = {'species': 0, 'genus': 1, 'family':2, 'order':3, 'class':4, 'phylum': 5}
# 	path_dl_toda_tax = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/data/dl_toda_taxonomy.tsv'
# 	print(path_dl_toda_tax)
# 	with open(path_dl_toda_tax, 'r') as in_f:
# 		content = in_f.readlines()
# 		dltoda_tax = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[1].split(';')[ranks_index[args.rank]] for line in content}

# 	# get information about testing reads mapping testing genome
# 	reads_id, pos_coverage, testing_genome_pos, alignment_reads_info = GetInfoTestingGenome(args)

#     # # get mean confidence scores at each position of the testing genome
# 	# confidence_scores = GetConfidenceScores(args, len(testing_genome_pos), alignment_reads_info, reads_id)

#    	# # get taxa mapped to each 
# 	# df_taxa = GetTaxa(args, dltoda_tax, testing_genome_pos, alignment_reads_info)
# 	# # df_taxa = pd.read_csv('/scratch/workspace/cecile_cres_uri_edu-dl-toda/dl-toda-bert/bin_read_classifiers/bbmap_analysis/711/taxa_read_count_711_df.csv')
# 	# # reset the index and remove first column
# 	# # new_index = [str(i) for i in range(1,df_taxa.shape[0]+1,1)]
# 	# # df_taxa  = df_taxa.set_index(pd.Index(new_index))
# 	# # df_taxa = df_taxa.iloc[:, 1:]
# 	# # count the number of columns with non zero values per row
# 	# df_taxa['UniqueTaxaCount'] = (df_taxa != 0).sum(axis=1)
# 	# unique_taxa_count = df_taxa['UniqueTaxaCount'].tolist()
# 	# # sum values in columns and sort columns based on sum
# 	# column_sums = df_taxa.sum(axis=0).sort_values()
# 	# new_df = pd.DataFrame()
# 	# new_df['sum'] = column_sums
# 	# new_df.to_csv(f'{args.rank}_sum_{args.label}_df.csv')
# 	# # sum values in rows
# 	# df_taxa['TotalTaxaCount'] = df_taxa.sum(axis=1)
# 	# total_taxa_count = df_taxa['TotalTaxaCount'].tolist()
# 	# print(type(total_taxa_count))
# 	# print(min(total_taxa_count))
# 	# print(max(total_taxa_count))
# 	# print(type(unique_taxa_count))
# 	# print(min(unique_taxa_count))
# 	# print(max(unique_taxa_count))
# 	# # create circos plot showing the testing genome and other info
# 	# PlotCircles(testing_genome_pos, unique_taxa_count, total_taxa_count, confidence_scores, pos_coverage, args.output_dir, args.label)

# 	# get files with results 
# 	files_w_results = glob.glob(os.path.join(args.input_dir, '*/*/*/*/testing-*/*_false_positives.tsv'))
# 	print(len(files_w_results))

# 	info_all = open(os.path.join(args.input_dir, 'false_positives_summary.tsv'), 'w')
# 	# info_selected = open(os.path.join(args.input_dir, 'false_positives_summary_above_40.tsv'), 'w')
# 	total_reads_pred = []
# 	relevant_reads = []
# 	relevant_labels = set()
# 	for input_file in files_w_results:
# 		pred_label = input_file.split('/')[-1].split('_')[1]
# 		df = pd.read_csv(input_file, sep='\t')
# 		confidence_scores = df['score'].to_list()
# 		label_reads_id = df['read_id'].to_list()
# 		reads_pred = (len(confidence_scores)/len(reads_id))*100
# 		total_reads_pred.append(reads_pred)
# 		# if reads_pred >= 40:
# 		# 	relevant_reads += df['read_id'].to_list()
# 		# 	info_selected.write(f'{pred_label}\t{dltoda_tax[pred_label]}\t{len(confidence_scores)}\t{reads_pred}\t{statistics.mean(confidence_scores)}\t{statistics.median(confidence_scores)}\t{min(confidence_scores)}\t{max(confidence_scores)}\n')
# 		info_all.write(f'{pred_label}\t{dltoda_tax[pred_label]}\t{reads_pred}\t{statistics.mean(confidence_scores)}\t{statistics.median(confidence_scores)}\t{min(confidence_scores)}\t{max(confidence_scores)}\n')
# 		for i in range(len(label_reads_id)):
# 			if float(confidence_scores[i]) >= 0.99:
# 				relevant_reads.append(label_reads_id[i])
# 				relevant_labels.add(pred_label)
# 	print(statistics.mean(total_reads_pred), min(total_reads_pred), max(total_reads_pred), statistics.median(total_reads_pred))
# 	plt.hist(total_reads_pred, bins=30)
# 	plt.xlabel('Fraction of testing reads predicted to be true')
# 	plt.ylabel('Frequency')
# 	plt.savefig(os.path.join(args.input_dir, 'false_positives.png'))

# 	count_relevant_reads = Counter(relevant_reads)
# 	count_relevant_reads_sorted = {k: v for k, v in sorted(count_relevant_reads.items(), key=lambda item: item[1])}


# 	print(len(relevant_reads))
# 	print(len(set(relevant_reads)))
# 	with open(os.path.join(args.input_dir, 'relevant_reads.tsv'), 'w') as f:
# 		json.dump(count_relevant_reads_sorted, f)

# 	with open(os.path.join(args.input_dir, 'relevant_labels.tsv'), 'w') as f:
# 		for l in list(relevant_labels):
# 			f.write(f'{l}\t{dltoda_tax[l]}\n')



# if __name__ == "__main__":
# 	main()