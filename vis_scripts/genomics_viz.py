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
from pygenomeviz.parser import Fasta
from pygenomeviz.utils import load_example_fasta_dataset, ColorCycler, interpolate_color
from pygenomeviz.align import AlignCoord, Blast
from matplotlib.patches import Patch
ColorCycler.set_cmap("Set1")

QUERY_TRACK_SIZE = 5
MIN_IDENTITY = 70
TICKS_INTERVAL = 100000

def GetAlignmentsInfo(sequences, samfile, sequence_length, seq_to_labels, output_dir, type):
	alignments = defaultdict(list)
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
						mapping_score = int(line.rstrip().split('\t')[4])
						alignments[read_id] = [seq_label, start_pos, start_pos+sequence_length[read_id], mapping_score]
						mapped_reads_id.append(read_id)
					else:
						unmapped_reads_id.append(read_id)
	
	with open(os.path.join(output_dir, f'{type}_mapped'), 'w') as f:
		f.write(''.join([f'{r}\n' for r in mapped_reads_id]))

	with open(os.path.join(output_dir, f'{type}_unmapped'), 'w') as f:
		f.write(''.join([f'{r}\n' for r in unmapped_reads_id]))

	return alignments, mapped_reads_id, unmapped_reads_id


def GetSeqLength(sequences_id, sequence_length, type):
	if len(sequences_id) != 0:
		seq_length_info = [sequence_length[s] for s in sequences_id]
		print(f'{type}\tmean: {statistics.mean(seq_length_info)}\tmedian: {statistics.median(seq_length_info)}\tmax: {max(seq_length_info)}\tmin: {min(seq_length_info)}')


def PlotCircos(args, fn_alignments_pos_test, fn_alignments_neg_train, tp_alignments_pos_test, test_test_coverage, train_test_coverage, output_filename, train_pos_coverage=None):

	# load data from training and testing genomes
	if train_pos_coverage:
		target_fasta = Fasta(args.train_fasta_239) # ref/subject
		queries = [args.test_fasta_239] # query
	else:
		target_fasta = Fasta(args.test_fasta_239) # ref/subject
		queries = [args.train_fasta_239, args.train_fasta_1308] # query
	
	comp_fasta_list = list(map(Fasta, queries))

	# Initialize circos instance
	circos = Circos(
	    sectors=target_fasta.get_seqid2size(),
	    # space=0 if len(target_fasta.get_seqid2size()) == 1 else 2,
	    space=5,
	)
	print('define space', len(target_fasta.get_seqid2size()))
	# circos.text(f"{target_fasta.name}\n({target_fasta.full_genome_length:,} bp)", size=13)
	print(f"{target_fasta.name}\n({target_fasta.full_genome_length:,} bp)")


	min_r_pos = 100
	for sector in circos.sectors:
		# plot genomic sector axis & xticks
		# track = sector.add_track((95, 100))
		# min_r_pos -= 0.3
		track = sector.add_track((min_r_pos - 0.3, min_r_pos))
		print(min_r_pos - 0.3, min_r_pos)
		track.axis(fc="black")
		if sector.size >= TICKS_INTERVAL:
			track.xticks_by_interval(
				TICKS_INTERVAL,
				# outer=False,
				label_formatter=lambda v: f"{v/1000000:.1f} Mb",
				label_orientation="vertical",
			)

	# Blast genome comparison & plot match blocks
	min_r_pos -= 0.3
	comp_name2color = {}
	colors = ["black", "gray"]
	for idx, comp_fasta in enumerate(comp_fasta_list):
	    align_coords = Blast([target_fasta, comp_fasta]).run()
	    align_coords = AlignCoord.filter(align_coords, identity_thr=MIN_IDENTITY)
	    # color = ColorCycler()
	    comp_name2color[comp_fasta.name] = colors[idx]
	    min_r_pos -= QUERY_TRACK_SIZE
	    print(min_r_pos, min_r_pos + QUERY_TRACK_SIZE)
	    for sector in circos.sectors:
	        sector.add_track((min_r_pos, min_r_pos + QUERY_TRACK_SIZE), r_pad_ratio=0.1)
	    for ac in align_coords:
	        track = circos.get_sector(ac.query_name).tracks[-1] # Last added track in sector
	        rect_color = interpolate_color(colors[idx], v=ac.identity, vmin=MIN_IDENTITY) # type: ignore
	        track.rect(ac.query_start, ac.query_end, color=rect_color)
	
	for sector in circos.sectors:
		min_r_pos -= 12
		# add track for TP reads
		tp_track = sector.add_track((min_r_pos, min_r_pos + 10), r_pad_ratio=0.1)
		for data in tp_alignments_pos_test.values():
			tp_track.rect(data[1], data[2], color="orange", lw=0.1)
		print(min_r_pos, min_r_pos + 10)
		print(f'added TP track')

		# add tracks for FN reads
		min_r_pos -= 12
		fn_track = sector.add_track((min_r_pos, min_r_pos + 10), r_pad_ratio=0.1)
		for read_id, data in fn_alignments_pos_test.items():
			fn_track.rect(data[1], data[2], color="red", lw=0.1)
		fn_track.axis(fc="white", ec="red")
		print(min_r_pos, min_r_pos + 10)
		print(f'added FN track')

		# add track for coverage of testing reads
		min_r_pos -= 12
		test_cov_track = sector.add_track((min_r_pos, min_r_pos + 10), r_pad_ratio=0.1)
		test_cov_track.axis()
		test_cov_track_y = list(range(min([int(i) for i in test_test_coverage]), max([math.ceil(j) for j in test_test_coverage])+1, 3))
		test_cov_track_y_labels = list(map(str, test_cov_track_y))
		test_cov_track.yticks(test_cov_track_y, test_cov_track_y_labels)
		test_cov_track.line([i for i in range(target_fasta.full_genome_length)], test_test_coverage, color="darkolivegreen")
		print(f'added test reads coverage track')

		# add track for coverage of testing reads
		min_r_pos -= 12
		train_cov_track = sector.add_track((min_r_pos, min_r_pos + 10), r_pad_ratio=0.1)
		train_cov_track.axis()
		train_cov_track_y = list(range(min([int(i) for i in train_test_coverage]), max([math.ceil(j) for j in train_test_coverage])+1, 2))
		train_cov_track_y_labels = list(map(str, train_cov_track_y))
		train_cov_track.yticks(train_cov_track_y, train_cov_track_y_labels)
		train_cov_track.line([i for i in range(target_fasta.full_genome_length)], train_test_coverage, color="yellowgreen")
		print(f'added train reads coverage track')

	# save figure
	circos.savefig(output_filename, dpi=300)
		


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--pos_test_neg_train', type=str, help='path to sam file with mapping of testing sequences from label 1 to training genomes from label 0 - get info about FN sequences')
	parser.add_argument('--pos_test_pos_test', type=str, help='path to sam file with mapping of testing sequences from label 1 to testing genome from label 1')
	parser.add_argument('--pos_train_pos_train', type=str, help='path to sam file with mapping of training sequences from 1 to training genome from label 1 - get info about coverage')
	parser.add_argument('--neg_test_pos_train', type=str, help='path to sam file with mapping of testing sequences from label 0 to training genome from label 1')
	parser.add_argument('--pos_train_fasta', type=str, help='path to training genome fasta file from label 1')
	parser.add_argument('--pos_test_fasta', type=str, help='path to testing genome fasta file from label 1')
	parser.add_argument('--annot_pos_train', type=str, help='annotations of label 1 training genome')
	parser.add_argument('--testing_fq_file', type=str, help='path to fastq file containing all testing reads (label 1 and 0)')
	parser.add_argument('--label', type=str, help='label of species investigated')
	parser.add_argument('--sequences_info', type=str, help='path to file mapping labels of species in model to sequences id of all sequences in training set')
	parser.add_argument('--prob_threshold', type=float, help='probability score threshold')
	parser.add_argument('--testing_genome_size', type=int, help='size in bp of genome of interest')
	parser.add_argument('--rank', type=str, help='taxonomic rank investigated', choices=['species','genus','family','order','class', 'phylum'])
	parser.add_argument('--testing_results', type=str, help='path to file containing testing results')
	parser.add_argument('--output_dir', type=str, help='path to output directory', default=os.getcwd())
	args = parser.parse_args()
	
	path_dl_toda_tax = '/'.join(
                os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/data/dl_toda_taxonomy.tsv'
	with open(path_dl_toda_tax, 'r') as in_f:
		content = in_f.readlines()
		args.dl_toda_tax = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[1] for line in content}

	# get reads in testing set fastq file
	sequences = load_fq_file(args.testing_fq_file, 4)
	sequence_length = {line.split('\n')[0][1:]: len(line.split('\n')[1]) for line in sequences}

	# get FN, FP and TP sequences
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
	GetSeqLength(list(fn_sequences), sequence_length, f'label {args.label} testing FN sequences')
	GetSeqLength(list(fp_sequences), sequence_length, f'label {args.label} testing FP sequences')
	GetSeqLength(list(tp_sequences), sequence_length, f'label {args.label} testing TP sequences')

	# get association between sequences in training set and labels
	with open(args.sequences_info, 'r') as f:
		content = f.readlines()
		seq_to_labels = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[1] for line in content}

	# # get alignments info for FN and TP reads
	# fn_alignments_neg_train, fn_mapped_reads_id_neg_train, fn_unmapped_reads_id_neg_train = GetAlignmentsInfo(fn_sequences, args.pos_test_neg_train, sequence_length, seq_to_labels, args.output_dir, 'false_negatives_pos_test_neg_train')
	# fn_alignments_pos_test, fn_mapped_reads_id_pos_test, fn_unmapped_reads_id_pos_test = GetAlignmentsInfo(fn_sequences, args.pos_test_pos_test, sequence_length, seq_to_labels, args.output_dir, 'false_negatives_pos_test_pos_test')
	# tp_alignments_pos_test, tp_mapped_reads_id_pos_test, tp_unmapped_reads_id_pos_test = GetAlignmentsInfo(tp_sequences, args.pos_test_pos_test, sequence_length, seq_to_labels, args.output_dir, 'true_positives_pos_test_pos_test')
	# print('FN - neg train', len(fn_alignments_neg_train), len(fn_mapped_reads_id_neg_train), len(fn_unmapped_reads_id_neg_train))
	# print('FN - pos test', len(fn_alignments_pos_test), len(fn_mapped_reads_id_pos_test), len(fn_unmapped_reads_id_pos_test))
	# print('TP - pos test', len(tp_alignments_pos_test), len(tp_mapped_reads_id_pos_test), len(tp_unmapped_reads_id_pos_test))
	# GetSeqLength(fn_mapped_reads_id_neg_train, sequence_length, 'label 239 mapped testing FN sequences to label 239 training genome')
	# GetSeqLength(fn_mapped_reads_id_pos_test, sequence_length, 'label 239 mapped testing FN sequences to label 239 testing genome')
	# GetSeqLength(fn_unmapped_reads_id_pos_test, sequence_length, 'label 239 unmapped testing FN sequences to label 239 testing genome')
	# GetSeqLength(fn_unmapped_reads_id_neg_train, sequence_length, 'label 239 unmapped testing FN sequences to label 239 training genome')
	# GetSeqLength(tp_mapped_reads_id_pos_test, sequence_length, 'label 239 mapped testing TP sequences to label 239 testing genome')
	# GetSeqLength(tp_unmapped_reads_id_pos_test, sequence_length, 'label 239 unmapped testing TP sequences to label 239 testing genome')

	# # get coverage of testing genome with testing sequences
	# test_ref_info, test_test_alignments = LoadData(args.pos_test_pos_test)
	# testing_genome_size = test_ref_info[0][1]
	# print(f'testing_genome_size: {testing_genome_size})')
	# test_test_dict_coverage, test_test_reads_info = GetCoverageOfSample(test_test_alignments[test_ref_info[0][0]], testing_genome_size, label=args.label)
	# test_test_coverage = [test_test_dict_coverage[i] for i in range(testing_genome_size)]
	# positions_w_zero = 0
	# for i in range(len(test_test_coverage)):
	# 	if test_test_coverage[i] == 0:
	# 		positions_w_zero += 1
	# print(f'test-test coverage: {positions_w_zero}')
	# print(f'mean: {statistics.mean(test_test_coverage)}\tmedian: {statistics.median(test_test_coverage)}\tmin: {min(test_test_coverage)}\tmax: {max(test_test_coverage)}')

	# # get coverage of testing genome with testing sequences
	# _, train_test_alignments = LoadData(args.pos_train_pos_test)
	# train_test_dict_coverage, train_test_reads_info = GetCoverageOfSample(train_test_alignments[test_ref_info[0][0]], testing_genome_size, label=args.label)
	# train_test_coverage = [train_test_dict_coverage[i] for i in range(testing_genome_size)]
	# positions_w_zero = 0
	# for i in range(len(train_test_coverage)):
	# 	if train_test_coverage[i] == 0:
	# 		positions_w_zero += 1
	# print(f'train-test coverage: {positions_w_zero}')
	# print(f'mean: {statistics.mean(train_test_coverage)}\tmedian: {statistics.median(train_test_coverage)}\tmin: {min(train_test_coverage)}\tmax: {max(train_test_coverage)}')

	# # create circos plot with FN reads info
	# PlotCircos(args, fn_alignments_pos_test, fn_alignments_neg_train, tp_alignments_pos_test, test_test_coverage, train_test_coverage, os.path.join(args.output_dir, f'{args.label}_false_negatives_test_genome.png'))

	# do FN analysis
	fn_alignments_neg_train, fn_mapped_reads_id_neg_train, fn_unmapped_reads_id_neg_train = GetAlignmentsInfo(fn_sequences, args.pos_test_neg_train, sequence_length, seq_to_labels, args.output_dir, 'false_negatives_pos_test_neg_train')

	# do FP analysis
	fp_taxa = defaultdict(int)
	for s in list(fp_sequences):
		taxon = args.dl_toda_tax[s.split('|')[1]]
		fp_taxa[taxon] += 1

	print(len(fp_taxa), statistics.mean(fp_taxa.values()),statistics.median(fp_taxa.values()), max(fp_taxa.values()), min(fp_taxa.values()))








