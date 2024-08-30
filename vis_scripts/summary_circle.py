import sys
import os
from pycirclize import Circos
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from dataprep_scripts.utils import load_fq_file
from collections import defaultdict
import random
import numpy as np
import math
import multiprocessing as mp


def sum_data(output_dir, label, mapped_pos_conf_scores, mapped_neg_conf_scores, mapped_neg_label, mapped_pos_label, all_conf_scores):
	# get average of cs
	with open(os.path.join(output_dir, f'{label}_pos_cs_mapped.tsv'), 'w') as f:
		for k, v in mapped_pos_label.items():
			mapped_pos_conf_scores[k] = mapped_pos_conf_scores[k]/v
			f.write(f'{k}\t{mapped_pos_conf_scores[k]}\n')

	with open(os.path.join(output_dir, f'{label}_neg_cs_mapped.tsv'), 'w') as f:
		for k, v in mapped_neg_label.items():
			mapped_neg_conf_scores[k] = mapped_neg_conf_scores[k]/v
			f.write(f'{k}\t{mapped_neg_conf_scores[k]}\n')

	# get percentages of positive and negative labels
	mapped_pos_label_percent = defaultdict(float)
	mapped_neg_label_percent = defaultdict(float)
	for i in range(ref_length):
		if i in mapped_neg_label and i in mapped_pos_label:
			mapped_pos_label_percent[i] = mapped_pos_label[i]/(mapped_pos_label[i]+mapped_neg_label[i])
			mapped_neg_label_percent[i] = mapped_neg_label[i]/(mapped_pos_label[i]+mapped_neg_label[i])
		if i in mapped_neg_label and i not in mapped_pos_label:
			mapped_neg_label_percent[i] = 1.0
		if i not in mapped_neg_label and i in mapped_pos_label:
			mapped_pos_label_percent[i] = 1.0

	outf_pos = open(os.path.join(output_dir, f'{label}_pos_label_mapped.tsv'), 'w')
	for k, v in mapped_pos_label_percent.items():
		outf_pos.write(f'{k}\t{v}\n')

	outf_neg = open(os.path.join(output_dir, f'{label}_neg_label_mapped.tsv'), 'w')
	for k, v in mapped_neg_label_percent.items():
		outf_neg.write(f'{k}\t{v}\n')

	with open(os.path.join(output_dir, f'{label}_all_conf_scores.tsv'), 'w') as f:
		for k, v in all_conf_scores.items():
			for i in range(len(v)):
				f.write(f'{k}\t{v[i]}\n')

	return mapped_pos_conf_scores, mapped_neg_conf_scores, mapped_pos_label_percent, mapped_neg_label_percent


def prep_test_results(output_dir, testing_output, alignment_sum, reads_id, label, ref_length):
	# get testing results
	with open(testing_output, 'r') as f:
		content = f.readlines()
		test_results = dict(zip(reads_id,[i.rstrip() for i in content]))

	# get alignment info
	with open(alignment_sum, 'r') as f:
		content = f.readlines()
		map_info = {i.rstrip().split('\t')[0]: int(i.rstrip().split('\t')[1]) for i in content}

	print(f'# test reads: {len(test_results)}\n # test reads mapped to training genome: {len(map_info)}')

	# initialize data structures to store info about reads from label of interest
	l_mapped_pos_label = defaultdict(int)
	l_mapped_neg_label = defaultdict(int)
	l_mapped_pos_conf_scores = defaultdict(float)
	l_mapped_neg_conf_scores = defaultdict(float)
	l_all_conf_scores = defaultdict(list)

	# initialize data structures to store info about reads from other labels
	o_mapped_pos_label = defaultdict(int)
	o_mapped_neg_label = defaultdict(int)
	o_mapped_pos_conf_scores = defaultdict(float)
	o_mapped_neg_conf_scores = defaultdict(float)
	o_all_conf_scores = defaultdict(list)

	for r in reads_id:
		# get predicted label and confidence score
		pred_label = int(test_results[r].rstrip().split('\t')[1])
		cs = float(test_results[r].rstrip().split('\t')[2])
		if r.split('|')[1] == label:
			# check if read mapped to the genome
			if r in map_info:
				# get start position where the read maps to the target genome
				start_pos = map_info[r] - 1
				print(f'{r}\t{start_pos}\t{pred_label}\t{cs}\t{start_pos + 250}')
				# add info
				for i in range(start_pos, start_pos + 250, 1):
					if pred_label == 1:
						l_mapped_pos_label[i] += 1
						l_mapped_pos_conf_scores[i] += cs
					elif pred_label == 0:
						l_mapped_neg_label[i] += 1
						l_mapped_neg_conf_scores[i] += cs
			l_all_conf_scores[pred_label].append(cs)
		else:
			# check if read mapped to the genome
			if r in map_info:
				# get start position where the read maps to the target genome
				start_pos = map_info[r] - 1
				print(f'{r}\t{start_pos}\t{pred_label}\t{cs}\t{start_pos + 250}')
				# add info
				for i in range(start_pos, start_pos + 250, 1):
					if pred_label == 1:
						o_mapped_pos_label[i] += 1
						o_mapped_pos_conf_scores[i] += cs
					elif pred_label == 0:
						o_mapped_neg_label[i] += 1
						o_mapped_neg_conf_scores[i] += cs
			l_all_conf_scores[pred_label].append(cs)

	l_mapped_pos_conf_scores, l_mapped_neg_conf_scores, l_mapped_pos_label_percent, l_mapped_neg_label_percent = sum_data(output_dir, label, l_mapped_pos_conf_scores, l_mapped_neg_conf_scores, l_mapped_neg_label, l_mapped_pos_label, l_all_conf_scores)
	o_mapped_pos_conf_scores, o_mapped_neg_conf_scores, o_mapped_pos_label_percent, o_mapped_neg_label_percent = sum_data(output_dir, 'other', l_mapped_pos_conf_scores, l_mapped_neg_conf_scores, l_mapped_neg_label, l_mapped_pos_label, l_all_conf_scores)

	return l_mapped_pos_conf_scores, l_mapped_neg_conf_scores, l_mapped_pos_label_percent, l_mapped_neg_label_percent, o_mapped_pos_conf_scores, o_mapped_neg_conf_scores, o_mapped_pos_label_percent, o_mapped_neg_label_percent


def plot_circles(output_dir, base_positions, test_pos_coverage, train_pos_coverage, pos_conf_scores, neg_conf_scores, pos_label, neg_label):
	# def plot_circles(output_dir, base_positions, pos_coverage, pos_conf_scores, neg_conf_scores, pos_label, neg_label, number):

	# initialize a single circos sector
	sectors = {'genome': len(base_positions)}
	circos = Circos(sectors=sectors, space=12)

	for sector in circos.sectors:
		# add outer track
		genome_track = sector.add_track((98, 100))
		genome_track.axis(fc="lightgrey")
		genome_x = list(range(0,len(base_positions),500000))
		base_pos_ticks = [base_positions[i] for i in genome_x]
		genome_x_labels = [f'{i/1000} Kb' for i in base_pos_ticks]
		genome_track.xticks(genome_x, genome_x_labels)
		genome_track.xticks_by_interval(100000, tick_length=1, show_label=False)
		print(f'added genome track')
		# add track for coverage of training reads
		print(len(base_positions), len(train_pos_coverage))
		cov_track = sector.add_track((85, 95))
		cov_track.axis()
		cov_y = list(range(min([int(i) for i in train_pos_coverage]), max([math.ceil(j) for j in train_pos_coverage])+1, 2))
		cov_y_labels = list(map(str, cov_y))
		cov_track.yticks(cov_y, cov_y_labels)
		cov_track.line(list(range(0,len(base_positions),1)), train_pos_coverage, color="#00A5E3")
		print(f'added coverage track')
		# add track for coverage of testing reads
		print(len(base_positions), len(test_pos_coverage))
		cov_track = sector.add_track((72, 82))
		cov_track.axis()
		cov_y = list(range(min([int(i) for i in test_pos_coverage]), max([math.ceil(j) for j in test_pos_coverage])+1, 2))
		cov_y_labels = list(map(str, cov_y))
		cov_track.yticks(cov_y, cov_y_labels)
		cov_track.line(list(range(0,len(base_positions),1)), test_pos_coverage, color="#8DD7BF")
		print(f'added coverage track')
		# add track for labels predicted as positive
		pos_labels_track = sector.add_track((59, 69))
		pos_labels_track.axis()
		pos_labels_y = [0.0, 0.5, 1.0]
		pos_labels_y_labels = list(map(str, pos_labels_y))
		pos_labels_track.yticks(pos_labels_y, pos_labels_y_labels)
		pos_labels_x = sorted(list(pos_label.keys()))
		pos_labels_x_values = [pos_label[i] for i in pos_labels_x]
		pos_labels_track.scatter(pos_labels_x, pos_labels_x_values, color="#FF96C5")
		print(f'added pos labels track')
		# add track for the confidence scores assigned to labels predicted as positive
		pos_cs_track = sector.add_track((46, 56))
		pos_cs_track.axis()
		pos_cs_y = [0.0, 0.5, 1.0]
		pos_cs_y_labels = list(map(str, pos_cs_y))
		pos_cs_track.yticks(pos_cs_y, pos_cs_y_labels)
		pos_cs_x = sorted(list(pos_conf_scores.keys()))
		pos_cs_x_values = [pos_conf_scores[i] for i in pos_cs_x]
		pos_cs_track.scatter(pos_cs_x, pos_cs_x_values, color="#FC6238")
		print(f'added pos cs track')
		# add track for labels predicted as negative
		neg_labels_track = sector.add_track((33, 43))
		neg_labels_track.axis()
		neg_labels_y = [0.0, 0.5, 1.0]
		neg_labels_y_labels = list(map(str, neg_labels_y))
		neg_labels_track.yticks(neg_labels_y, neg_labels_y_labels)
		neg_labels_x = sorted(list(neg_label.keys()))
		neg_labels_x_values = [neg_label[i] for i in neg_labels_x]
		neg_labels_track.scatter(neg_labels_x, neg_labels_x_values, color="#FF5768")
		print(f'added neg labels track')
		# add track for the confidence scores assigned to labels predicted as negative
		neg_cs_track = sector.add_track((20, 30))
		neg_cs_track.axis()
		neg_cs_y = [0.0, 0.5, 1.0]
		neg_cs_y_labels = list(map(str, neg_cs_y))
		neg_cs_track.yticks(neg_cs_y, neg_cs_y_labels)
		neg_cs_x = sorted(list(neg_conf_scores.keys()))
		neg_cs_x_values = [neg_conf_scores[i] for i in neg_cs_x]
		neg_cs_track.scatter(neg_cs_x, neg_cs_x_values, color="#FFBF65")
		print(f'added neg cs track')

	# circos.savefig(os.path.join(output_dir, f'sum_circos_{number}.png'))
	circos.savefig(os.path.join(output_dir, f'sum_circos.png'))


def get_coverage(filename):
	with open(filename, 'r') as f:
		content = f.readlines()
		pos_coverage = [math.log(int(i.rstrip().split('\t')[1])) if int(i.rstrip().split('\t')[1]) != 0 else 0.0 for i in content]
		base_positions = list(range(0,len(pos_coverage),1))

	return pos_coverage, base_positions


def main():
	fq_file = sys.argv[1]
	testing_output = sys.argv[2]
	train_reads_genome_cov = sys.argv[3]
	test_reads_genome_cov = sys.argv[4]
	alignment_sum = sys.argv[5]
	label = sys.argv[6]
	output_dir = sys.argv[7]

	# get reads id
	num_lines = 4
	reads = load_fq_file(fq_file, num_lines)
	reads_id = [r.split("\n")[0][1:] for r in reads]
	print(f'# reads: {len(reads_id)}\t{len(reads)}\t{reads_id[0]}')

	# load coverage of training reads to training genome
	train_pos_coverage, base_positions = get_coverage(train_reads_genome_cov)
	# load coverage of testing reads to training genome
	test_pos_coverage, _ = get_coverage(test_reads_genome_cov) 
	# load testing results
	l_mapped_pos_conf_scores, l_mapped_neg_conf_scores, l_mapped_pos_label_percent, l_mapped_neg_label_percent, \
	o_mapped_pos_conf_scores, o_mapped_neg_conf_scores, o_mapped_pos_label_percent, o_mapped_neg_label_percent = prep_test_results(output_dir, testing_output, \
		alignment_sum, reads_id, label, len(base_positions))

	print(f'{len(test_pos_coverage)}\t{len(train_pos_coverage)}')
	
	# create plots
	plot_circles(label, output_dir, base_positions, test_pos_coverage, train_pos_coverage, l_mapped_pos_conf_scores, l_mapped_neg_conf_scores, l_mapped_pos_label, l_mapped_neg_label)
	plot_circles(label, output_dir, base_positions, test_pos_coverage, train_pos_coverage, o_mapped_pos_conf_scores, o_mapped_neg_conf_scores, o_mapped_pos_label, o_mapped_neg_label)






if __name__ == "__main__":
	main()