import sys
import os
import glob
import json
from pycirclize import Circos
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from dataprep_scripts.utils import load_fq_file
from vis_scripts.parse_samfile import LoadData, GetCoverage
from collections import defaultdict
import random
import numpy as np
import math
import pandas as pd
import multiprocessing as mp
import argparse
import statistics
from matplotlib import colormaps
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from collections import Counter

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


def GetMappedReadsInfo(args, samfile, fqfile):
	# load data about alignments of testing reads to testing genome
	ref_info, mapped, unmapped = LoadData(samfile)
	print(ref_info)
	print(len(mapped))
	print(len(unmapped))
	
	assert len(ref_info) == 1, f'{samfile} has more than 1 reference sequence'
	
	# select alignments for reads from label of interest
	label_mapped = [a for a in mapped[ref_info[0][0]] if a[0].split('|')[1] == args.label]
	print(len(label_mapped))

	genome_length = ref_info[0][1]
	genome_pos = list(range(1, testing_genome_length+1, 1))
	dict_coverage, reads_info = GetCoverage(label_mapped, genome_length)
	pos_coverage = [dict_coverage[i] for i in range(genome_length)]

	reads = load_fq_file(fqfile, 4)
	reads_id = []
	dict_reads_length = {}
	for r in reads:
		read_id = r.split("\n")[0][1:]
		length = len(r.split("\n")[1])
		if r.split("\n")[0].split('|')[1] == args.label:
			dict_reads_length[read_id] = length
			reads_id.append(read_id)
	print(len(reads_id))

	# get unmapped reads and their length
	label_unmapped = set(list(dict_reads_length.keys())).difference(set(list(reads_info.keys())))
	with open(os.path.join(args.output_dir, f'unmapped_reads_{args.label}.tsv'), 'w') as f:
		for r in label_unmapped:
			read_label = r.split('|')[1]
			if read_label == args.label:
				f.write(f'{r}\t{dict_reads_length[r]}\n')

	return pos_coverage


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--samfiles', type=str, help='path to directory containing SAM files')
	parser.add_argument('--fqfiles', type=str, help='path to directory containing fq files')
	parser.add_argument('--label', type=str, help='label of species investigated')
	parser.add_argument('--output_dir', type=str, help='path to output directory', default=os.getcwd())
	args = parser.parse_args()

	# get information about testing reads mapping testing genome
	test_pos_coverage = GetMappedReadsInfo(args, os.path.join(args.samfiles, 'testing_data_vs_testing_genome_results.sam'), os.path.join(args.fqfiles, f'finetuning_l{args.label}_test_data_k4_cleaned.fq'))
	# # get information about training reads mapping training genome
	# train_ref_info, train_alignments = GetMappedReadsInfo(os.path.join(args.samfiles, 'training_data_vs_training_genome_results.sam'))
	# # get information about testing reads mapping training genome
	# _, test_train_alignments = GetMappedReadsInfo(os.path.join(args.samfiles, 'testing_data_vs_training_genome_results.sam'))




if __name__ == "__main__":
	main()