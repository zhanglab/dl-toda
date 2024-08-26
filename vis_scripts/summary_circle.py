import sys
import os
from pycirclize import Circos
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from dataprep_scripts.utils import load_fq_file
from collections import defaultdict
import random
import numpy as np
import math

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

	pos_label = [0]*ref_length # number of reads given the positive label at every position in the genome
	neg_label = [0]*ref_length
	pos_conf_scores = [0.0]*ref_length
	neg_conf_scores = [0.0]*ref_length
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
	for i in range(ref_length):
		if pos_label[i] != 0:
			pos_conf_scores[i] = round(pos_conf_scores[i]/pos_label[i], 2)
		if neg_label[i] != 0:
			neg_conf_scores[i] = round(neg_conf_scores[i]/neg_label[i], 2)

	# get percentages of positive and negative labels
	pos_label_percent = []
	neg_label_percent = []
	for i in range(ref_length):
		if pos_label[i] + neg_label[i] != 0:
			pos_label_percent.append(pos_label[i]/(pos_label[i]+neg_label[i]))
			neg_label_percent.append(neg_label[i]/(pos_label[i]+neg_label[i]))
		else:
			pos_label_percent.append(0)
			neg_label_percent.append(0)

	return pos_label_percent, neg_label_percent, pos_conf_scores, neg_conf_scores


def plot_circles(base_positions, pos_coverage, pos_conf_scores, neg_conf_scores, pos_label, neg_label, number):
	# define x axis
	# base_positions = list(range(0,len(pos_coverage),1))
	# initialize a single circos sector
	sectors = {'genome': len(pos_coverage)}
	circos = Circos(sectors=sectors, space=10)

	# define min and max y values
	cs_min = min(pos_conf_scores+neg_conf_scores)
	cs_max = max(pos_conf_scores+neg_conf_scores)
	labels_min = min(pos_label+neg_label)
	labels_max = max(pos_label+neg_label)
	print(f'cs_min: {cs_min}\tcs_max: {cs_max}\tlabels_min: {labels_min}\tlabels_max: {labels_max}')
	for sector in circos.sectors:
		# add outer track
		genome_track = sector.add_track((98, 100))
		genome_track.axis(fc="lightgrey")
		genome_track.xticks_by_interval(100000, label_formatter=lambda v: f"{v / 1000:.0f} Kb")
		genome_track.xticks_by_interval(1000, tick_length=1, show_label=False)
		print(f'added genome track')
		# add track for coverage
		cov_track = sector.add_track((85, 95))
		cov_track.axis()
		cov_y = list(range(min([int(i) for i in pos_coverage]), max([math.ceil(j) for j in pos_coverage])+1, 1))
		cov_y_labels = list(map(str, cov_y))
		cov_track.yticks(cov_y, cov_y_labels)
		cov_track.line(base_positions, pos_coverage, color="0")
		print(f'added coverage track')
		# add track for labels predicted as positive
		pos_labels_track = sector.add_track((70, 80))
		pos_labels_track.axis()
		pos_labels_y = [x / 10.0 for x in range(0, 10+3, 3)]
		pos_labels_y_labels = list(map(str, pos_labels_y))
		pos_labels_track.yticks(pos_labels_y, pos_labels_y_labels)
		pos_labels_track.line(base_positions, pos_label, color="#BF2C34")
		# pos_labels_track.heatmap(pos_label, vmin=labels_min, vmax=labels_max, show_value=False)
		print(f'added pos labels track')
		# add track for the confidence scores assigned to labels predicted as positive
		pos_cs_track = sector.add_track((55, 65))
		pos_cs_track.axis()
		pos_cs_y = [x / 10.0 for x in range(0, 10+3, 3)]
		pos_cs_y_labels = list(map(str, pos_cs_y))
		pos_cs_track.yticks(pos_cs_y, pos_cs_y_labels)
		pos_cs_track.scatter(base_positions, pos_conf_scores, color="#SC62D6")
		# pos_cs_track.heatmap(pos_conf_scores, vmin=cs_min, vmax=cs_max, show_value=False)
		print(f'added pos cs track')
		# add track for labels predicted as negative
		neg_labels_track = sector.add_track((40, 50))
		neg_labels_track.axis()
		neg_labels_y = [x / 10.0 for x in range(0, 10+3, 3)]
		neg_labels_y_labels = list(map(str, neg_labels_y))
		neg_labels_track.yticks(neg_labels_y, neg_labels_y_labels)
		neg_labels_track.line(base_positions, neg_label, color="#BE398D")
		print(f'added neg labels track')
		# add track for the confidence scores assigned to labels predicted as negative
		neg_cs_track = sector.add_track((25, 35))
		neg_cs_track.axis()
		neg_cs_y = [x / 10.0 for x in range(0, 10+3, 3)]
		neg_cs_y_labels = list(map(str, neg_cs_y))
		neg_cs_track.yticks(neg_cs_y, neg_cs_y_labels)
		neg_cs_track.scatter(base_positions, neg_conf_scores, color="#F07857")
		print(f'added neg cs track')
	
	# circos.colorbar(bounds=(0.35, 0.55, 0.3, 0.01), vmin=labels_min, vmax=labels_max, orientation="horizontal", cmap="viridis")
	# circos.colorbar(bounds=(0.35, 0.45, 0.3, 0.01), vmin=cs_min, vmax=cs_max, orientation="horizontal", cmap="plasma")

	circos.savefig(os.path.join(output_dir, f'sum_circos_{number}.png'))


def main():
	fq_file = sys.argv[1]
	testing_output = sys.argv[2]
	genome_cov = sys.argv[3]
	alignment_sum = sys.argv[4]
	label = sys.argv[5]
	output_dir = sys.argv[6]

	# get reads id
	num_lines = 4
	reads = load_fq_file(fq_file, num_lines)
	reads_id = [r.split("\n")[0][1:] for r in reads]
	print(f'# reads: {len(reads_id)}\t{len(reads)}\t{reads_id[0]}')

	# get coverage of target genome per position
	with open(genome_cov, 'r') as f:
		content = f.readlines()
		pos_coverage = [math.log(int(i.rstrip().split('\t')[1])) if int(i.rstrip().split('\t')[1]) != 0 else 0.0 for i in content]
		base_positions = list(range(0,len(pos_coverage),1))

	pos_label, neg_label, pos_conf_scores, neg_conf_scores = prep_test_results(testing_output, alignment_sum, reads_id, label, len(pos_coverage))

	print(f'{len(pos_coverage)}\t{len(pos_label)}\t{len(neg_label)}\t{len(pos_conf_scores)}\t{len(neg_conf_scores)}')
	print(f'{pos_label[:10]}\n{pos_conf_scores[:10]}\n{neg_label[:10]}\n{neg_conf_scores[:10]}\n{pos_coverage[:10]}')
	print(f'sum positive labels: {sum(pos_label)}\t sum negative labels: {sum(neg_label)}')
	print(f'labels_min: {min(pos_label)}\tlabels_max: {max(pos_label)}')
	print(f'labels_min: {min(neg_label)}\tlabels_max: {max(neg_label)}')
	print(f'cov_min: {min(pos_coverage)}\tcov_max: {max(pos_coverage)}')

	# divide data into 5 subsets and create a circle plot for each subset
	subset_size = math.ceil(len(pos_coverage)/6)
	pos_label_subsets = [pos_label[i:i+subset_size] for i in range(0, len(pos_label), subset_size)]
	neg_label_subsets = [neg_label[i:i+subset_size] for i in range(0, len(neg_label), subset_size)]
	pos_cs_subsets = [pos_conf_scores[i:i+subset_size] for i in range(0, len(pos_conf_scores), subset_size)]
	neg_cs_subsets = [neg_conf_scores[i:i+subset_size] for i in range(0, len(neg_conf_scores), subset_size)]
	pos_cov_subsets = [pos_coverage[i:i+subset_size] for i in range(0, len(pos_coverage), subset_size)]
	base_pos_subsets = [base_positions[i:i+subset_size] for i in range(0, len(base_positions), subset_size)]

	total = 0
	for i in range(6):
		print(base_pos_subsets[i][0], base_pos_subsets[i][-1])
		print(len(base_pos_subsets[i]), len(pos_cov_subsets[i]))
		total += len(base_pos_subsets[i])
		plot_circles(base_pos_subsets[i], pos_cov_subsets[i], pos_cs_subsets[i], neg_cs_subsets[i], pos_label_subsets[i], pos_label_subsets[i], i)
	print(total)
	




if __name__ == "__main__":
	main()