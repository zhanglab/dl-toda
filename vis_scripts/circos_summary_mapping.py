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

def CreatePlot(pos_coverage, figname):
	
	# initialize a single circos sector
	sectors = {'genome': len(pos_coverage)}
	circos = Circos(sectors=sectors, space=14)

	genome_positions = list(range(len(pos_coverage)))
	print(len(genome_positions), genome_positions[0], genome_positions[-1])

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
		cov_y = list(range(min([int(i) for i in pos_coverage]), max([math.ceil(j) for j in pos_coverage])+1, 4))
		cov_y_labels = list(map(str, cov_y))
		cov_track.yticks(cov_y, cov_y_labels)
		cov_track.line(list(range(0,len(genome_positions),1)), pos_coverage, color="#00A5E3")
		print(f'added coverage track')
		# save figure
		circos.savefig(figname)


def GetMappedReadsInfo(args, samfile, fqfile):
	# load data about alignments of testing reads to testing genome
	ref_info, mapped = LoadData(samfile)
	print(ref_info)
	print(f'mapped: {len(mapped)}')
	
	assert len(ref_info) == 1, f'{samfile} has more than 1 reference sequence'
	
	# select alignments for reads from label of interest
	label_mapped = [a for a in mapped[ref_info[0][0]] if a[0].split('|')[1] == args.label]
	print(f'label_mapped: {len(label_mapped)}')

	genome_length = ref_info[0][1]
	genome_pos = list(range(1, genome_length+1, 1))
	dict_coverage, reads_info = GetCoverage(label_mapped, genome_length)
	pos_coverage = [dict_coverage[i] for i in range(genome_length)]
	print(genome_length, len(pos_coverage), pos_coverage[:10])

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

	with open(os.path.join(args.output_dir, f'coverage-info.tsv'), 'w') as f:
		f.write(f'min: {min(pos_coverage)}\nmax: {max(pos_coverage)}\nmean: {statistics.mean(pos_coverage)}\nmedian: {statistics.median(pos_coverage)}')

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
	CreatePlot(test_pos_coverage, os.path.join(args.output_dir, f'test_test_{args.label}.png'))
	# get information about training reads mapping training genome
	train_pos_coverage = GetMappedReadsInfo(args, os.path.join(args.samfiles, 'training_data_vs_training_genome_results.sam'), os.path.join(args.fqfiles, f'finetuning_l{args.label}_train_data_k4_cleaned.fq'))
	CreatePlot(train_pos_coverage, os.path.join(args.output_dir, f'train_train_{args.label}.png'))
	# get information about testing reads mapping training genome
	test_train_pos_coverage = GetMappedReadsInfo(args, os.path.join(args.samfiles, 'testing_data_vs_training_genome_results.sam'), os.path.join(args.fqfiles, f'finetuning_l{args.label}_test_data_k4_cleaned.fq'))
	CreatePlot(test_train_pos_coverage, os.path.join(args.output_dir, f'test_train_{args.label}.png'))



if __name__ == "__main__":
	main()