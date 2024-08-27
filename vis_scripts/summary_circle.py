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

def prep_test_results(testing_output, alignment_sum, reads_id, label, ref_length):
	# get testing results
	with open(testing_output, 'r') as f:
		content = f.readlines()
		test_results = dict(zip(reads_id,[i.rstrip() for i in content]))

	# get alignment info
	with open(alignment_sum, 'r') as f:
		content = f.readlines()
		map_info = {i.rstrip().split('\t')[0]: int(i.rstrip().split('\t')[1]) for i in content}

	print(f'# test reads: {len(test_results)}\n # test reads mapped to training genome: {len(map_info)}')

	pos_label = defaultdict(int)
	neg_label = defaultdict(int)
	pos_conf_scores = defaultdict(float)
	neg_conf_scores = defaultdict(float)

	for r in reads_id:
		if r.split('|')[1] == label:
			# check if read mapped to the genome
			if r in map_info:
				# get start position where the read maps to the target genome
				start_pos = map_info[r] - 1
				# get predicted label
				pred_label = int(test_results[r].rstrip().split('\t')[1])
				cs = float(test_results[r].rstrip().split('\t')[2])
				print(f'{r}\t{start_pos}\t{pred_label}\t{cs}\t{start_pos + 250}')
				# add info
				for i in range(start_pos, start_pos + 250, 1):
					if pred_label == 1:
						pos_label[i] += 1
						pos_conf_scores[i] += cs
					elif pred_label == 0:
						neg_label[i] += 1
						neg_conf_scores[i] += cs

	# get average of cs
	for k, v in pos_label.items():
		pos_conf_scores[k] = pos_conf_scores[k]/v

	for k, v in neg_label.items():
		neg_conf_scores[k] = neg_conf_scores[k]/v

	# get percentages of positive and negative labels
	pos_label_percent = defaultdict(float)
	neg_label_percent = defaultdict(float)
	for i in range(ref_length):
		if i in neg_label and i in pos_label:
			pos_label_percent[i] = pos_label[i]/(pos_label[i]+neg_label[i])
			neg_label_percent[i] = neg_label[i]/(pos_label[i]+neg_label[i])
		if i in neg_label and i not in pos_label:
			neg_label_percent[i] = 1.0
		if i not in neg_label and i in pos_label:
			pos_label_percent[i] = 1.0

	return pos_label_percent, neg_label_percent, pos_conf_scores, neg_conf_scores


def plot_circles(output_dir, base_positions, test_pos_coverage, train_pos_coverage, pos_conf_scores, neg_conf_scores, pos_label, neg_label):
	# def plot_circles(output_dir, base_positions, pos_coverage, pos_conf_scores, neg_conf_scores, pos_label, neg_label, number):

	# initialize a single circos sector
	sectors = {'genome': len(base_positions)}
	circos = Circos(sectors=sectors, space=10)

	for sector in circos.sectors:
		# add outer track
		genome_track = sector.add_track((98, 100))
		genome_track.axis(fc="lightgrey")
		genome_x = list(range(0,len(base_positions),10000))
		base_pos_ticks = [base_positions[i] for i in genome_x]
		genome_x_labels = [str(i) for i in base_pos_ticks]
		genome_track.xticks(genome_x, genome_x_labels)
		genome_track.xticks_by_interval(1000, tick_length=1, show_label=False)
		print(f'added genome track')
		# add track for coverage of training reads
		print(len(base_positions), len(train_pos_coverage))
		cov_track = sector.add_track((87, 97))
		cov_track.axis()
		cov_y = list(range(min([int(i) for i in train_pos_coverage]), max([math.ceil(j) for j in train_pos_coverage])+1, 2))
		cov_y_labels = list(map(str, cov_y))
		cov_track.yticks(cov_y, cov_y_labels)
		cov_track.line(list(range(0,len(base_positions),1)), train_pos_coverage, color="#CFF800")
		print(f'added coverage track')
		# add track for coverage of testing reads
		print(len(base_positions), len(test_pos_coverage))
		cov_track = sector.add_track((76, 86))
		cov_track.axis()
		cov_y = list(range(min([int(i) for i in test_pos_coverage]), max([math.ceil(j) for j in test_pos_coverage])+1, 2))
		cov_y_labels = list(map(str, cov_y))
		cov_track.yticks(cov_y, cov_y_labels)
		cov_track.line(list(range(0,len(base_positions),1)), test_pos_coverage, color="#C05780")
		print(f'added coverage track')
		# add track for labels predicted as positive
		pos_labels_track = sector.add_track((65, 75))
		pos_labels_track.axis()
		pos_labels_y = [0.0, 0.5, 1.0]
		pos_labels_y_labels = list(map(str, pos_labels_y))
		pos_labels_track.yticks(pos_labels_y, pos_labels_y_labels)
		pos_labels_x = sorted(list(pos_label.keys()))
		pos_labels_x_values = [pos_label[i] for i in pos_labels_x]
		pos_labels_track.line(pos_labels_x, pos_labels_x_values, color="#FF828B")
		print(f'added pos labels track')
		# add track for the confidence scores assigned to labels predicted as positive
		pos_cs_track = sector.add_track((54, 64))
		pos_cs_track.axis()
		pos_cs_y = [0.0, 0.5, 1.0]
		pos_cs_y_labels = list(map(str, pos_cs_y))
		pos_cs_track.yticks(pos_cs_y, pos_cs_y_labels)
		pos_cs_x = sorted(list(pos_conf_scores.keys()))
		pos_cs_x_values = [pos_conf_scores[i] for i in pos_cs_x]
		pos_cs_track.scatter(pos_cs_x, pos_cs_x_values, color="#E7C582")
		print(f'added pos cs track')
		# add track for labels predicted as negative
		neg_labels_track = sector.add_track((43, 53))
		neg_labels_track.axis()
		neg_labels_y = [0.0, 0.5, 1.0]
		neg_labels_y_labels = list(map(str, neg_labels_y))
		neg_labels_track.yticks(neg_labels_y, neg_labels_y_labels)
		neg_labels_x = sorted(list(neg_label.keys()))
		neg_labels_x_values = [neg_label[i] for i in neg_labels_x]
		neg_labels_track.line(neg_labels_x, neg_labels_x_values, color="#00B0BA")
		print(f'added neg labels track')
		# add track for the confidence scores assigned to labels predicted as negative
		neg_cs_track = sector.add_track((32, 42))
		neg_cs_track.axis()
		neg_cs_y = [0.0, 0.5, 1.0]
		neg_cs_y_labels = list(map(str, neg_cs_y))
		neg_cs_track.yticks(neg_cs_y, neg_cs_y_labels)
		neg_cs_x = sorted(list(neg_conf_scores.keys()))
		neg_cs_x_values = [neg_conf_scores[i] for i in neg_cs_x]
		neg_cs_track.bar(neg_cs_x, neg_cs_x_values, color="#0065A2")
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
	pos_label, neg_label, pos_conf_scores, neg_conf_scores = prep_test_results(testing_output, alignment_sum, reads_id, label, len(base_positions))

	print(f'{len(test_pos_coverage)}\t{len(train_pos_coverage)}\t{len(pos_label)}\t{len(neg_label)}\t{len(pos_conf_scores)}\t{len(neg_conf_scores)}')
	
	# # divide data into 5 subsets and create a circle plot for each subset
	# num_subsets = 20
	# subset_size = math.ceil(len(pos_coverage)/num_subsets)
	# pos_label_subsets = [pos_label[i:i+subset_size] for i in range(0, len(pos_label), subset_size)]
	# neg_label_subsets = [neg_label[i:i+subset_size] for i in range(0, len(neg_label), subset_size)]
	# pos_cs_subsets = [pos_conf_scores[i:i+subset_size] for i in range(0, len(pos_conf_scores), subset_size)]
	# neg_cs_subsets = [neg_conf_scores[i:i+subset_size] for i in range(0, len(neg_conf_scores), subset_size)]
	# pos_cov_subsets = [pos_coverage[i:i+subset_size] for i in range(0, len(pos_coverage), subset_size)]
	# base_pos_subsets = [base_positions[i:i+subset_size] for i in range(0, len(base_positions), subset_size)]
	# with mp.Manager() as manager:
	# 	processes = [mp.Process(target=plot_circles, args=(output_dir, base_pos_subsets[i], pos_cov_subsets[i], pos_cs_subsets[i], neg_cs_subsets[i], pos_label_subsets[i], neg_label_subsets[i], i)) for i in range(num_subsets)]
	# 	for p in processes:
	# 		p.start()
	# 	for p in processes:
	# 		p.join()

	plot_circles(output_dir, base_positions, test_pos_coverage, train_pos_coverage, pos_conf_scores, neg_conf_scores, pos_label, neg_label)

	




if __name__ == "__main__":
	main()