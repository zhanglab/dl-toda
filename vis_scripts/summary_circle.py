import sys
import os
import glob
from pycirclize import Circos
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from dataprep_scripts.utils import load_fq_file
from vis_scripts.parse_samfile import load_data
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

# def sum_data(output_dir, label, mapped_pos_conf_scores, mapped_neg_conf_scores, mapped_neg_label, mapped_pos_label, all_conf_scores, ref_length):
# 	# get average of cs
# 	with open(os.path.join(output_dir, f'{label}_pos_cs_mapped.tsv'), 'w') as f:
# 		for k, v in mapped_pos_label.items():
# 			mapped_pos_conf_scores[k] = mapped_pos_conf_scores[k]/v
# 			f.write(f'{k}\t{mapped_pos_conf_scores[k]}\n')

# 	with open(os.path.join(output_dir, f'{label}_neg_cs_mapped.tsv'), 'w') as f:
# 		for k, v in mapped_neg_label.items():
# 			mapped_neg_conf_scores[k] = mapped_neg_conf_scores[k]/v
# 			f.write(f'{k}\t{mapped_neg_conf_scores[k]}\n')

# 	# get percentages of positive and negative labels
# 	mapped_pos_label_percent = defaultdict(float)
# 	mapped_neg_label_percent = defaultdict(float)
# 	for i in range(ref_length):
# 		if i in mapped_neg_label and i in mapped_pos_label:
# 			mapped_pos_label_percent[i] = mapped_pos_label[i]/(mapped_pos_label[i]+mapped_neg_label[i])
# 			mapped_neg_label_percent[i] = mapped_neg_label[i]/(mapped_pos_label[i]+mapped_neg_label[i])
# 		if i in mapped_neg_label and i not in mapped_pos_label:
# 			mapped_neg_label_percent[i] = 1.0
# 		if i not in mapped_neg_label and i in mapped_pos_label:
# 			mapped_pos_label_percent[i] = 1.0

# 	outf_pos = open(os.path.join(output_dir, f'{label}_pos_label_mapped.tsv'), 'w')
# 	for k, v in mapped_pos_label_percent.items():
# 		outf_pos.write(f'{k}\t{v}\n')

# 	outf_neg = open(os.path.join(output_dir, f'{label}_neg_label_mapped.tsv'), 'w')
# 	for k, v in mapped_neg_label_percent.items():
# 		outf_neg.write(f'{k}\t{v}\n')

# 	with open(os.path.join(output_dir, f'{label}_all_conf_scores.tsv'), 'w') as f:
# 		for k, v in all_conf_scores.items():
# 			for i in range(len(v)):
# 				f.write(f'{k}\t{v[i]}\n')

# 	return mapped_pos_conf_scores, mapped_neg_conf_scores, mapped_pos_label_percent, mapped_neg_label_percent


# def prep_test_results(output_dir, testing_output, alignment_sum, reads_id, label, ref_length):
# 	# get testing results
# 	with open(testing_output, 'r') as f:
# 		content = f.readlines()
# 		test_results = dict(zip(reads_id,[i.rstrip() for i in content]))

# 	# get alignment info
# 	with open(alignment_sum, 'r') as f:
# 		content = f.readlines()
# 		map_info = {i.rstrip().split('\t')[0]: int(i.rstrip().split('\t')[1]) for i in content}

# 	print(f'# test reads: {len(test_results)}\n # test reads mapped to training genome: {len(map_info)}')

# 	# initialize data structures to store info about reads from label of interest
# 	l_mapped_pos_label = defaultdict(int)
# 	l_mapped_neg_label = defaultdict(int)
# 	l_mapped_pos_conf_scores = defaultdict(float)
# 	l_mapped_neg_conf_scores = defaultdict(float)
# 	l_all_conf_scores = defaultdict(list)

# 	# initialize data structures to store info about reads from other labels
# 	o_mapped_pos_label = defaultdict(int)
# 	o_mapped_neg_label = defaultdict(int)
# 	o_mapped_pos_conf_scores = defaultdict(float)
# 	o_mapped_neg_conf_scores = defaultdict(float)
# 	o_all_conf_scores = defaultdict(list)

# 	for r in reads_id:
# 		# get predicted label and confidence score
# 		pred_label = int(test_results[r].rstrip().split('\t')[1])
# 		cs = float(test_results[r].rstrip().split('\t')[2])
# 		if r.split('|')[1] == label:
# 			# check if read mapped to the genome
# 			if r in map_info:
# 				# get start position where the read maps to the target genome
# 				start_pos = map_info[r] - 1
# 				print(f'{r}\t{start_pos}\t{pred_label}\t{cs}\t{start_pos + 250}')
# 				# add info
# 				for i in range(start_pos, start_pos + 250, 1):
# 					if pred_label == 1:
# 						l_mapped_pos_label[i] += 1
# 						l_mapped_pos_conf_scores[i] += cs
# 					elif pred_label == 0:
# 						l_mapped_neg_label[i] += 1
# 						l_mapped_neg_conf_scores[i] += cs
# 			l_all_conf_scores[pred_label].append(cs)
# 		else:
# 			# check if read mapped to the genome
# 			if r in map_info:
# 				# get start position where the read maps to the target genome
# 				start_pos = map_info[r] - 1
# 				print(f'{r}\t{start_pos}\t{pred_label}\t{cs}\t{start_pos + 250}')
# 				# add info
# 				for i in range(start_pos, start_pos + 250, 1):
# 					if pred_label == 1:
# 						o_mapped_pos_label[i] += 1
# 						o_mapped_pos_conf_scores[i] += cs
# 					elif pred_label == 0:
# 						o_mapped_neg_label[i] += 1
# 						o_mapped_neg_conf_scores[i] += cs
# 			l_all_conf_scores[pred_label].append(cs)

# 	l_mapped_pos_conf_scores, l_mapped_neg_conf_scores, l_mapped_pos_label_percent, l_mapped_neg_label_percent = sum_data(output_dir, label, l_mapped_pos_conf_scores, l_mapped_neg_conf_scores, l_mapped_neg_label, l_mapped_pos_label, l_all_conf_scores, ref_length)
# 	o_mapped_pos_conf_scores, o_mapped_neg_conf_scores, o_mapped_pos_label_percent, o_mapped_neg_label_percent = sum_data(output_dir, 'other', l_mapped_pos_conf_scores, l_mapped_neg_conf_scores, l_mapped_neg_label, l_mapped_pos_label, l_all_conf_scores, ref_length)

# 	return l_mapped_pos_conf_scores, l_mapped_neg_conf_scores, l_mapped_pos_label_percent, l_mapped_neg_label_percent, o_mapped_pos_conf_scores, o_mapped_neg_conf_scores, o_mapped_pos_label_percent, o_mapped_neg_label_percent


def PlotCircles(genome_positions, unique_taxa_count, total_taxa_count, confidence_scores, pos_coverage, output_dir, label):
	
	# initialize a single circos sector
	sectors = {'genome': len(genome_positions)}
	circos = Circos(sectors=sectors, space=14)

	for sector in circos.sectors:
		# add track for positions of the label's testing genome
		genome_track = sector.add_track((98, 100))
		genome_track.axis(fc="lightgrey")
		genome_x = list(range(0,len(genome_positions),500000))
		base_pos_ticks = [genome_positions[i] for i in genome_x]
		genome_x_labels = [f'{i/1000} Kb' for i in base_pos_ticks]
		genome_track.xticks(genome_x, genome_x_labels)
		genome_track.xticks_by_interval(100000, tick_length=1, show_label=False)
		print(f'added genome track')
		# add track for coverage of label's training genome
		cov_track = sector.add_track((85, 95))
		cov_track.axis()
		cov_y = list(range(min([int(i) for i in pos_coverage]), max([math.ceil(j) for j in pos_coverage])+1, 2))
		cov_y_labels = list(map(str, cov_y))
		cov_track.yticks(cov_y, cov_y_labels)
		cov_track.line(list(range(0,len(genome_positions),1)), pos_coverage, color="#00A5E3")
		print(f'added coverage track')
		# add track for confidence scores obtained of the label's testing reads
		pos_cs_track = sector.add_track((72, 82))
		pos_cs_track.axis()
		pos_cs_y = [0.0, 0.5, 1.0]
		pos_cs_y_labels = list(map(str, pos_cs_y))
		pos_cs_track.yticks(pos_cs_y, pos_cs_y_labels)
		pos_cs_x = genome_positions
		pos_cs_x_values = [confidence_scores[i] for i in pos_cs_x]
		pos_cs_track.line(pos_cs_x, pos_cs_x_values, color="#FC6238")
		print(f'added pos cs track')
		# add track for count of unique taxa per position
		unique_taxa_track = sector.add_track((59, 69))
		unique_taxa_track.axis()
		unique_taxa_y = list(range(min(unique_taxa_count), max(unique_taxa_count)+1, 10))
		unique_taxa_y_labels = list(map(str, unique_taxa_y))
		unique_taxa_track.yticks(unique_taxa_y, unique_taxa_y_labels)
		unique_taxa_x = genome_positions
		unique_taxa_x_values = unique_taxa_count
		unique_taxa_track.line(unique_taxa_x, unique_taxa_x_values, color="#9e1369")
		# add track for count of total taxa per position
		total_taxa_track = sector.add_track((46, 56))
		total_taxa_track.axis()
		total_taxa_y = list(range(int(min(total_taxa_count)), int(max(total_taxa_count))+1, 1000))
		print(total_taxa_y)
		total_taxa_y_labels = list(map(str, total_taxa_y))
		print(total_taxa_y_labels)
		total_taxa_track.yticks(total_taxa_y, total_taxa_y_labels)
		total_taxa_x = genome_positions
		total_taxa_x_values = total_taxa_count
		total_taxa_track.line(total_taxa_x, total_taxa_x_values, color="#465d66")
		# save figure
		circos.savefig(os.path.join(output_dir, f'circos_{label}.png'))


# def plot_circles(label, output_dir, base_positions, test_pos_coverage, train_pos_coverage, pos_conf_scores, neg_conf_scores, pos_label, neg_label):
# 	# def plot_circles(output_dir, base_positions, pos_coverage, pos_conf_scores, neg_conf_scores, pos_label, neg_label, number):

# 	# initialize a single circos sector
# 	sectors = {'genome': len(base_positions)}
# 	circos = Circos(sectors=sectors, space=14)

# 	for sector in circos.sectors:
# 		# add outer track
# 		genome_track = sector.add_track((98, 100))
# 		genome_track.axis(fc="lightgrey")
# 		genome_x = list(range(0,len(base_positions),500000))
# 		base_pos_ticks = [base_positions[i] for i in genome_x]
# 		genome_x_labels = [f'{i/1000} Kb' for i in base_pos_ticks]
# 		genome_track.xticks(genome_x, genome_x_labels)
# 		genome_track.xticks_by_interval(100000, tick_length=1, show_label=False)
# 		print(f'added genome track')
		# # add track for coverage of training reads
		# # print(len(base_positions), len(train_pos_coverage))
		# cov_track = sector.add_track((85, 95))
		# cov_track.axis()
		# cov_y = list(range(min([int(i) for i in train_pos_coverage]), max([math.ceil(j) for j in train_pos_coverage])+1, 2))
		# cov_y_labels = list(map(str, cov_y))
		# cov_track.yticks(cov_y, cov_y_labels)
		# cov_track.line(list(range(0,len(base_positions),1)),
		# add track for labels predicted as positive
		# pos_labels_track = sector.add_track((59, 69))
		# pos_labels_track.axis()
		# pos_labels_y = [0.0, 0.5, 1.0]
		# pos_labels_y_labels = list(map(str, pos_labels_y))
		# pos_labels_track.yticks(pos_labels_y, pos_labels_y_labels)
		# pos_labels_x = sorted(list(pos_label.keys()))
		# pos_labels_x_values = [pos_label[i] for i in pos_labels_x]
		# pos_labels_track.scatter(pos_labels_x, pos_labels_x_values, color="#FF96C5")
		# print(f'added pos labels track')
		# add track for the confidence scores assigned to labels predicted as positive
		# pos_cs_track = sector.add_track((46, 56))
		# pos_cs_track.axis()
		# pos_cs_y = [0.0, 0.5, 1.0]
		# pos_cs_y_labels = list(map(str, pos_cs_y))
		# pos_cs_track.yticks(pos_cs_y, pos_cs_y_labels)
		# pos_cs_x = sorted(list(pos_conf_scores.keys()))
		# pos_cs_x_values = [pos_conf_scores[i] for i in pos_cs_x]
		# pos_cs_track.scatter(pos_cs_x, pos_cs_x_values, color="#FC6238")
		# print(f'added pos cs track')
		# add track for labels predicted as negative
		# neg_labels_track = sector.add_track((33, 43))
		# neg_labels_track.axis()
		# neg_labels_y = [0.0, 0.5, 1.0]
		# neg_labels_y_labels = list(map(str, neg_labels_y))
		# neg_labels_track.yticks(neg_labels_y, neg_labels_y_labels)
		# neg_labels_x = sorted(list(neg_label.keys()))
		# neg_labels_x_values = [neg_label[i] for i in neg_labels_x]
		# neg_labels_track.scatter(neg_labels_x, neg_labels_x_values, color="#FF5768")
		# print(f'added neg labels track')
		# add track for the confidence scores assigned to labels predicted as negative
	# 	neg_cs_track = sector.add_track((20, 30))
	# 	neg_cs_track.axis()
	# 	neg_cs_y = [0.0, 0.5, 1.0]
	# 	neg_cs_y_labels = list(map(str, neg_cs_y))
	# 	neg_cs_track.yticks(neg_cs_y, neg_cs_y_labels)
	# 	neg_cs_x = sorted(list(neg_conf_scores.keys()))
	# 	neg_cs_x_values = [neg_conf_scores[i] for i in neg_cs_x]
	# 	neg_cs_track.scatter(neg_cs_x, neg_cs_x_values, color="#FFBF65")
	# 	print(f'added neg cs track')

	# # circos.savefig(os.path.join(output_dir, f'sum_circos_{number}.png'))
	# circos.savefig(os.path.join(output_dir, f'sum_circos_{label}.png'))


def GetTaxa(args, dltoda_tax, genome_positions, mapping_info):
	# load data about alignments of testing reads to training genomes
	samfiles = glob.glob(os.path.join(args.train_samfiles, '*.sam'))
	samfiles.remove(os.path.join(args.train_samfiles, f'{args.label}_results.sam'))
	mapped_species = defaultdict(set) # key = position in testing genome, value = list of taxa with a training genome to which the read was mapped to
	unique_mapped_taxa = set()
	for sam in samfiles:
		sam_label = sam.rstrip().split('/')[-1].split('_')[0]
		_, _, reads_info = load_data(args, sam)
		for k in reads_info.keys():
			# get start and end positions of alignment on the testing genome
			if k in mapping_info:
				start_position = mapping_info[k][0]
				end_position = mapping_info[k][1]
				for p in range(start_position, end_position+1, 1):
					mapped_species[p].add(sam_label)
					unique_mapped_taxa.add(dltoda_tax[sam_label])


    # create dataframe with rows = positions in testin genome and columns = mapped taxa
	unique_mapped_taxa = list(unique_mapped_taxa)
	row_num = len(genome_positions)
	col_num = len(unique_mapped_taxa)
    
	matrix = np.zeros((row_num, col_num))
	row_names = [str(i) for i in range(1,row_num+1,1)]
	col_names = [i for i in unique_mapped_taxa]
	df = pd.DataFrame(matrix, index=row_names, columns=col_names)
	for pos, taxa_list in mapped_species.items():
		for t in taxa_list:
			df.loc[pos, t] += 1
	print(df)
	df.to_csv(os.path.join(args.output_dir, f'taxa_read_count_{args.label}_df.csv'), index=False)
	return df


def GetCoverage(filename):
	with open(filename, 'r') as f:
		content = f.readlines()
		pos_coverage = [math.log(int(i.rstrip().split('\t')[1])) if int(i.rstrip().split('\t')[1]) != 0 else 0.0 for i in content]
		genome_positions = list(range(0,len(pos_coverage),1))

	return pos_coverage, genome_positions


def GetConfidenceScores(args, testing_genome_length, mapping_info, reads_id):
	# load testing results
	dict_confidence_scores = defaultdict(list)
    
	with open(args.testing_results, 'r') as f:
		content = f.readlines()
		testing_results_data = [line.rstrip().split('\t')[2] for line in content]
    
	assert len(testing_results_data) == len(reads_id), "the number of reads id does not match the number of reads tested"

	for i, r in enumerate(reads_id):
		label = r.split('|')[1]
    	# only get confidence scores of reads belonging to label
		if label == args.label:
			if r in mapping_info:
				start_pos = mapping_info[r][0]
				end_pos = mapping_info[r][1]
				confidence_score = float(testing_results_data[i])
				for i in range(start_pos, end_pos, 1):
					dict_confidence_scores[i].append(confidence_score)

	confidence_scores = []	
	for i in range(testing_genome_length):
		if i in dict_confidence_scores:
			confidence_scores.append(statistics.mean(dict_confidence_scores[i]))
		else:
			confidence_scores.append(0.0)

	return confidence_scores


def UnmappedReads(args, mapping_info):
	# load fq file with testing reads --> required to identify reads that were not mapped to the reference testing genome (too short)
	reads = load_fq_file(args.fq_file, 4)
	reads_id = []
	dict_reads_length = {}
	for r in reads:
		read_id = r.split("\n")[0][1:]
		length = len(r.split("\n")[1])
		dict_reads_length[read_id] = length
		reads_id.append(read_id)

	# get reads that were not mapped to the testing genome and their length
	unmapped_reads = set(list(dict_reads_length.keys())).difference(set(list(mapping_info.keys())))
	with open(os.path.join(args.output_dir, f'unmapped_reads_{args.label}.tsv'), 'w') as f:
		for r in unmapped_reads:
			read_label = r.split('|')[1]
			if read_label == args.label:
				f.write(f'{r}\t{dict_reads_length[r]}\n')

	return reads_id


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_samfiles', type=str, help='directory containing SAM files with alignment of testing reads to training genomes')
	parser.add_argument('--test_samfile', type=str, help='path to SAM file containing the alignment of testing reads to the testing genome')
	parser.add_argument('--test_coverage', type=str, help='path to file *-cov-pos.tsv containing number of mapped reads at each position')
	parser.add_argument('--fq_file', type=str, help='path to fastq file containing all testing reads (+ and - class)')
	parser.add_argument('--label', type=str, help='label of species investigated')
	parser.add_argument('--rank', type=str, help='taxonomic rank investigated', choices=['species','genus','family','order','class', 'phylum'])
	parser.add_argument('--testing_results', type=str, help='path to file containing testing results')
	parser.add_argument('--output_dir', type=str, help='path to output directory', default=os.getcwd())
	parser.add_argument('--num_processes', type=int, default=8)
	args = parser.parse_args()

    # define path to taxonomy of genomes in dltoda
	ranks_index = {'species': 0, 'genus': 1, 'family':2, 'order':3, 'class':4, 'phylum': 5}
	path_dl_toda_tax = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/data/dl_toda_taxonomy.tsv'
	print(os.path.dirname(os.path.abspath(__file__)))
	with open(path_dl_toda_tax, 'r') as in_f:
		content = in_f.readlines()
		dltoda_tax = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[1].split(';')[ranks_index[args.rank]] for line in content}


	# load data about alignments of testing reads to testing genome
	ref_info, _, mapping_info = load_data(args, args.test_samfile)
	testing_genome_length = ref_info[0][1]

    # get information about reads not mapped to testing genome and ordered list of reads id from all reads (+ and - classes)
	reads_id = UnmappedReads(args, mapping_info)

    # get mean confidence scores at each position of the testing genome
	confidence_scores = GetConfidenceScores(args, testing_genome_length, mapping_info, reads_id)

    # get coverage of training genome
	pos_coverage, genome_positions = GetCoverage(args.test_coverage)

   	# get taxa mapped to each 
	df_taxa = GetTaxa(args, dltoda_tax, genome_positions, mapping_info)
	# df_taxa = pd.read_csv('/scratch/workspace/cecile_cres_uri_edu-dl-toda/dl-toda-bert/bin_read_classifiers/bbmap_analysis/711/taxa_read_count_711_df.csv')
	# reset the index and remove first column
	# new_index = [str(i) for i in range(1,df_taxa.shape[0]+1,1)]
	# df_taxa  = df_taxa.set_index(pd.Index(new_index))
	# df_taxa = df_taxa.iloc[:, 1:]
	# count the number of columns with non zero values per row
	df_taxa['UniqueTaxaCount'] = (df_taxa != 0).sum(axis=1)
	unique_taxa_count = df_taxa['UniqueTaxaCount'].tolist()
	# sum values in columns and sort columns based on sum
	column_sums = df_taxa.sum(axis=0).sort_values()
	new_df = pd.DataFrame()
	new_df['sum'] = column_sums
	new_df.to_csv(f'{args.rank}_sum_{args.label}_df.csv')
	# sum values in rows
	df_taxa['TotalTaxaCount'] = df_taxa.sum(axis=1)
	total_taxa_count = df_taxa['TotalTaxaCount'].tolist()
	print(type(total_taxa_count))
	print(min(total_taxa_count))
	print(max(total_taxa_count))
	print(type(unique_taxa_count))
	print(min(unique_taxa_count))
	print(max(unique_taxa_count))
	# create circos plot showing the testing genome and other info
	PlotCircles(genome_positions, unique_taxa_count, total_taxa_count, confidence_scores, pos_coverage, args.output_dir, args.label)

	


	# # load coverage of training reads to training genome
	# train_pos_coverage, base_positions = get_coverage(train_reads_genome_cov)
	# # load coverage of testing reads to training genome
	# test_pos_coverage, _ = get_coverage(test_reads_genome_cov) 
	# # load testing results
	# l_mapped_pos_conf_scores, l_mapped_neg_conf_scores, l_mapped_pos_label_percent, l_mapped_neg_label_percent, \
	# o_mapped_pos_conf_scores, o_mapped_neg_conf_scores, o_mapped_pos_label_percent, o_mapped_neg_label_percent = prep_test_results(output_dir, testing_output, \
	# 	alignment_sum, reads_id, label, len(base_positions))

	# print(f'{len(test_pos_coverage)}\t{len(train_pos_coverage)}')
	
	# # create plots
	# plot_circles(label, output_dir, base_positions, test_pos_coverage, train_pos_coverage, l_mapped_pos_conf_scores, l_mapped_neg_conf_scores, l_mapped_pos_label_percent, l_mapped_neg_label_percent)
	# plot_circles('other', output_dir, base_positions, test_pos_coverage, train_pos_coverage, o_mapped_pos_conf_scores, o_mapped_neg_conf_scores, o_mapped_pos_label_percent, o_mapped_neg_label_percent)






if __name__ == "__main__":
	main()