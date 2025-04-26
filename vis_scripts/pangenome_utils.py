import sys
import os
import glob
import math
import zipfile
import subprocess
import random
import statistics
import numpy as np
from Bio import SeqIO, SeqUtils
from collections import defaultdict
from pycirclize import Circos, config
from pygenomeviz.parser import Fasta
from pygenomeviz.utils import load_example_fasta_dataset, ColorCycler, interpolate_color
from pygenomeviz.align import AlignCoord, Blast
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

ColorCycler.set_cmap("Set1")

# QUERY_TRACK_SIZE = 5
MIN_IDENTITY = 70
TICKS_INTERVAL = 500000
blastn_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/blastn"
makeblastdb_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/makeblastdb"
ncbi_datasets_exec = "/work/pi_yingzhang_uri_edu/ccres/tools/datasets"

# set seed
seed = 42

# set the global python random seed
random.seed(seed)


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



def GetMatchRegions(args, input_file, identity_thr=MIN_IDENTITY):
	align_coords = []
	query_pident = {}
	with open(input_file, 'r') as f:
		for count, line in enumerate(f, 1):
			sstart = int(line.rstrip().split(',')[2])
			send = int(line.rstrip().split(',')[3])
			qstart = int(line.rstrip().split(',')[4])
			qend = int(line.rstrip().split(',')[5])
			pident = float(line.rstrip().split(',')[8])
			qseq = line.rstrip().split(',')[9]
			sseq = line.rstrip().split(',')[10]
			for i in range(qstart, qend+1, 1):
				query_pident[i] = pident

			if pident >= identity_thr:
				align_coords.append([qstart, qend, pident])

	return align_coords, query_pident


def RunBlast(output_dir, query, num_processes, subject=None, db=False, outfilename=None, sam=False):
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
	if db:
		sys.executable = blastn_exec
		process = subprocess.run([sys.executable, '-query', f'{query}', '-db', '/datasets/bio/ncbi-db/2025-01-26/nt', '-out', \
			f'{output_dir}/blast/test_fp_blastn.out', '-outfmt', "10 delim=, qseqid sseqid evalue pident sstart send qstart qend length ssciname stitle", \
			'-max_target_seqs', '1', '-num_threads', f'{num_processes}'])
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
			 	'-outfmt', "17", '-max_target_seqs', '1', '-num_threads', f'{num_processes}'])
		else:
			result = subprocess.run([blastn_exec, '-query', f'{query}', '-db', f'{output_dir}/blastdb', '-out', f'{outfilename}', \
			 '-outfmt', "10 delim=, qseqid sseqid sstart send qstart qend qlen evalue pident qseq sseq sstrand", \
			 '-max_target_seqs', '5', '-num_threads', f'{num_processes}'])

def GetGenomesInfo(fasta):
	with open(fasta, 'r') as f:
		content = f.readline()
	strain = ' '.join([e for e in content.rstrip().split(',')[0].split(' ')[1:] if e not in ['chromosome', 'strain']])
	
	return strain


avg_pct_identity, ani, test_strain, train_strain = CircosPlot(args, scores, testing_records[0].seq, training_records[0].seq, args.testing_fasta, args.training_fasta, incorrect_alignments, correct_alignments, \
			 incorrect_genes, correct_genes, gene_to_incorrect_reads_kept, gene_to_correct_reads_kept, test_readid_to_read, os.path.join(args.output_dir, f'{args.testing_genome}_{args.prob_threshold}_circos.png'))


def CircosPlot(args, scores, test_record_seq, train_record_seq, testing_fasta, training_fasta, \
			incorrect_alignments, correct_alignments, incorrect_genes, correct_genes, gene_to_incorrect_reads_kept, gene_to_correct_reads_kept, test_readid_to_read, outfigpath):

	# load data from training and testing genomes of label 1
	query_fasta = Fasta(testing_fasta) # query --> testing genome
	ref_fasta = Fasta(training_fasta) # ref/subject --> training genome

	# Initialize circos instance
	circos = Circos(
	    sectors=query_fasta.get_seqid2size(),
	    # space=0 if len(ref_fasta.get_seqid2size()) == 1 else 2,
		space=10,
	)

	train_strain = GetGenomesInfo(training_fasta)
	test_strain = GetGenomesInfo(testing_fasta)
	circos.text(f'{test_strain}\n{query_fasta.full_genome_length:,} bp\n(testing genome)', size=9, r=22)

	with open(os.path.join(args.output_dir, f'{args.testing_genome}_genomes_length.tsv'), 'w') as f:
		f.write(f'Label 0 testing genome:\t{query_fasta.name}\t{query_fasta.full_genome_length}\n')
		f.write(f'Label 1 training genome:\t{ref_fasta.name}\t{ref_fasta.full_genome_length}\n')

	min_r_pos = 100
	for sector in circos.sectors:
		# Setup outer track
		outer_track = sector.add_track((min_r_pos-0.3, min_r_pos))
		outer_track.axis(fc="black")
		outer_track.xticks_by_interval(TICKS_INTERVAL, label_formatter=lambda v: f"{v/1000000:.1f} Mb",)
		min_r_pos -= 1
		outer_track.xticks_by_interval(100000, tick_length=1, show_label=False)

	# Blast genome comparison & plot match blocks
	# store percentage identity between matching regions
	percent_identity = []
	# run blast 	
	print('output dir', os.path.join(args.output_dir, 'blast', args.testing_genome, 'test_train_genomes'))	
	RunBlast(args, os.path.join(args.output_dir, 'blast', args.testing_genome, 'test_train_genomes'), testing_fasta, subject=[training_fasta], outfilename=f'{args.output_dir}/blast/{args.testing_genome}/test_train_genomes/test_train_genomes_blastn.out')
	align_coords, query_pident = GetMatchRegions(args, f'{args.output_dir}/blast/{args.testing_genome}/test_train_genomes/test_train_genomes_blastn.out', identity_thr=MIN_IDENTITY)

	# get average percentage identity per gene
	outf = open(os.path.join(args.output_dir, 'testing_reads_pident_prediction_attentions.tsv'), 'w')
	with open(os.path.join(args.output_dir, 'testing_genes_pident_training_genome.tsv'), 'w') as f:
		for gene_id, gene_info in correct_genes.items():
			gene_start = gene_info[5]
			gene_end = gene_info[6]
			pident_pos = []
			for i in range(gene_start, gene_end+1, 1):
				if i in query_pident:
					pident_pos.append(query_pident[i])
				else:
					pident_pos.append(0)
			avg_pident = round(sum(pident_pos)/len(pident_pos),3)
			f.write(f'{gene_id}\t{avg_pident}\tcorrect\n')
			if avg_pident >= 0.95:
				similarity = 'similar'
			else:
				similarity = 'dissimilar'
			list_reads = gene_to_correct_reads_kept[gene_id]
			for i in range(len(list_reads)):
				read_id = list_reads[i] + f'{gene_id}-correct-{similarity}'
				outf.write(f'{read_id}\t{test_readid_to_read[list_reads[i]]}\n')

		for gene_id, gene_info in incorrect_genes.items():
			gene_start = gene_info[5]
			gene_end = gene_info[6]
			pident_pos = []
			for i in range(gene_start, gene_end+1, 1):
				if i in query_pident:
					pident_pos.append(query_pident[i])
				else:
					pident_pos.append(0)
			avg_pident = round(sum(pident_pos)/len(pident_pos),3)
			f.write(f'{gene_id}\t{avg_pident}\tincorrect\n')
			if avg_pident >= 0.95:
				similarity = 'similar'
			else:
				similarity = 'dissimilar'
			list_reads = gene_to_incorrect_reads_kept[gene_id]
			for i in range(len(list_reads)):
				read_id = list_reads[i] + f'{gene_id}-incorrect-{similarity}'
				outf.write(f'{read_id}\t{test_readid_to_read[list_reads[i]]}\n')
	outf.close()

	# count the number of identical positions across the aligned regions
	identical_positions = 0
	for sector in circos.sectors:
		blast_track = sector.add_track((min_r_pos-5, min_r_pos), r_pad_ratio=0.1)
		min_r_pos -= 5	
		for ac in align_coords:
			percent_identity.append(ac[2])
			identical_positions += (ac[2]/100*(ac[1]-ac[0]))
			rect_color = interpolate_color("black", v=ac[2], vmin=MIN_IDENTITY)
			blast_track.rect(ac[0], ac[1], color=rect_color)

	# get stats on percentage identity
	avg_pct_identity = round(identical_positions/query_fasta.full_genome_length*100,2)
	ani = round(statistics.mean(percent_identity), 2)
	with open(os.path.join(args.output_dir, f'{args.testing_genome}_pct_identity_matching_regions.tsv'), 'w') as f:
		f.write(f'# identical positions\t{identical_positions}\npercentage identity\t{avg_pct_identity}%\n')
		f.write(f'Stats on aligned regions\nmean\t{statistics.mean(percent_identity)}\nmedian\t{statistics.median(percent_identity)}\nmin\t{min(percent_identity)}\nmax\t{max(percent_identity)}')

	for sector in circos.sectors:
		# define x-axis vector for the next tracks
		genome_pos = list(range(query_fasta.full_genome_length))

		# add track for scores
		min_r_pos -= 5
		scores_track = sector.add_track((min_r_pos-10, min_r_pos), r_pad_ratio=0.1)
		scores_track.axis(ec="deeppink")
		y_values = list(range(math.floor(min(scores)), math.ceil(max(scores))+1, 1))
		y_labels = list(map(str, y_values))
		scores_track.yticks(y_values, y_labels)
		scores_track.line(genome_pos, scores, color="deeppink")
		print(f'added score track')

		# add track for correct reads
		if len(correct_alignments) > 0:
			min_r_pos -= 13
			correct_track = sector.add_track((min_r_pos-10, min_r_pos), r_pad_ratio=0.1)
			correct_track.axis(ec="blue")
			pos_correct_count = [0]*query_fasta.full_genome_length
			for readid, data in correct_alignments.items():
				for pos in range(data[1], data[2]+1, 1):
					pos_correct_count[pos-1] +=1
			y_values = list(range(min(pos_correct_count), max(pos_correct_count), 1))
			y_labels = list(map(str, y_values))
			correct_track.yticks(y_values, y_labels)
			correct_track.line(genome_pos, pos_correct_count, color="blue")
			print(f'added correct track')

		# add tracks for incorrect reads 
		if len(incorrect_alignments) > 0:
			min_r_pos -= 13
			incorrect_track = sector.add_track((min_r_pos-10, min_r_pos), r_pad_ratio=0.1)
			incorrect_track.axis(ec="darkviolet")
			pos_incorrect_count = [0]*query_fasta.full_genome_length
			for readid, data in incorrect_alignments.items():
				for pos in range(data[1], data[2]+1, 1):
					pos_incorrect_count[pos-1] +=1
			y_values = list(range(min(pos_incorrect_count), max(pos_incorrect_count), 1))
			y_labels = list(map(str, y_values))
			incorrect_track.yticks(y_values, y_labels)
			incorrect_track.line(genome_pos, pos_incorrect_count, color="darkviolet")
			print(f'added incorrect track')

		# # Plot GC skew
		# min_r_pos -= 11
		# gcskew_track = sector.add_track((min_r_pos-5, min_r_pos))
		# pos_list, gcskews = GetGCSkew(test_record_seq)
		# positive_gcskews = np.where(gcskews > 0, gcskews, 0)
		# negative_gcskews = np.where(gcskews < 0, gcskews, 0)
		# abs_max_gcskew = np.max(np.abs(gcskews))
		# vmin, vmax = -abs_max_gcskew, abs_max_gcskew
		# gcskew_track.fill_between(
		# 	pos_list, positive_gcskews, 0, vmin=vmin, vmax=vmax, color="grey"
		# )
		# gcskew_track.fill_between(
		# 	pos_list, negative_gcskews, 0, vmin=vmin, vmax=vmax, color="limegreen"
		# )

		# # Plot GC content
		# min_r_pos -= 5
		# gc_content_track = sector.add_track((min_r_pos-5, min_r_pos))
		# pos_list, gc_content, test_genome_gc_content = GetGCContent(test_record_seq)
		# gc_content_updated = gc_content - test_genome_gc_content
		# positive_gc_content = np.where(gc_content_updated > 0, gc_content_updated, 0)
		# negative_gc_content = np.where(gc_content_updated < 0, gc_content_updated, 0)
		# abs_max_gc_content = np.max(np.abs(gc_content_updated))
		# vmin, vmax = -abs_max_gc_content, abs_max_gc_content
		# gc_content_track.fill_between(
		# 	pos_list, positive_gc_content, 0, vmin=vmin, vmax=vmax, color="black"
		# )
		# gc_content_track.fill_between(
		# 	pos_list, negative_gc_content, 0, vmin=vmin, vmax=vmax, color="deeppink"
		# )
		
		# # report GC content of train and test genomes
		# _, _, train_genome_gc_content = GetGCContent(train_record_seq)
		# with open(os.path.join(args.output_dir, f'{args.testing_genome}_GC_content.tsv'), 'w') as f:
		# 	f.write(f'Testing genome:\t{test_genome_gc_content}\n')
		# 	f.write(f'Training genome:\t{train_genome_gc_content}')

	# Save figure
	# Enable annotation text adjustment (Default)
	# config.ann_adjust.enable = True
	fig = circos.plotfig()
	# Add legend
	handles = []
	handles += [
		Patch(color='black', label=f'{train_strain}\n{ref_fasta.full_genome_length:,} bp (training genome) - {avg_pct_identity}% - {ani}%'),
		Patch(color='deeppink', label='Score')
	]
	if len(correct_alignments) > 0:
		handles.append(Patch(color='blue', label='True Positives'))
	if len(incorrect_alignments) > 0:
		handles.append(Patch(color='darkviolet', label='False Positives'))
		
	# handles += [
	# 	Line2D([], [], color='blue', label='Positive GC Skew', marker="^", ms=6, ls="None"),
	# 	Line2D([], [], color='gold', label='Negative GC Skew', marker="v", ms=6, ls="None"),
	# 	Line2D([], [], color='darkviolet', label='Positive GC Content', marker="^", ms=6, ls="None"),
	# 	Line2D([], [], color='orangered', label='Negative GC Content', marker="v", ms=6, ls="None")
	# 	]
	_ = circos.ax.legend(handles=handles, bbox_to_anchor=(0.5, 0.475), loc="center", fontsize=8)
	fig.savefig(outfigpath, dpi=300)

	return avg_pct_identity, ani, test_strain, train_strain


def CreateTsvFile(reads_id, readid_to_read, filename):	
	with open(filename, 'w') as f:
		for read_id, gene_id in reads_id.items():
			f.write(f'{read_id}-{gene_id}\t{readid_to_read[read_id]}\n')
			# f.write(''.join([f'>{r}\n{readid_to_read[r]}\n' for r in reads_id]))

def WriteInputAttentions(tsv_filename, id_filename, reads, reads_id, test_readid_to_read):
	tsv_file = open(tsv_filename, 'w')
	id_file = open(id_filename, 'w')
	for r in reads:
		id_file.write(f'{r[0]}')
		for idx in range(1, len(r), 1):
			id_file.write(f'\t{r[idx]}')
		id_file.write('\n')
	for k, v in reads_id.items():
		tsv_file.write(f'{v}\t{test_readid_to_read[k]}\n')
	tsv_file.close()
	id_file.close()


def ParseAlignments(alignments):
	reads_id = []
	start = []
	end = []
	strand = []
	for readid, data in alignments.items():
		if data[1] < data[2]:
			start_pos = data[1]
			end_pos = data[2]
		else:
			start_pos = data[2]
			end_pos = data[1]
		reads_id.append(readid)
		start.append(start_pos)
		end.append(end_pos)
		strand.append(data[5])	
	return reads_id, start, end, strand



def GetReadsForAttentions(args, correct_alignments, incorrect_alignments, incorrect_reads_kept, correct_reads_kept, test_readid_to_read):
	
	incorrect_reads_id, incorrect_start, incorrect_end, incorrect_strand = ParseAlignments(incorrect_alignments)
	correct_reads_id, correct_start, correct_end, correct_strand = ParseAlignments(correct_alignments)

	# get all reads
	all_reads = []
	all_reads_id = {}
	for i in range(len(incorrect_reads_id)):
		if incorrect_reads_id[i] in incorrect_reads_kept:
			all_reads.append([incorrect_reads_id[i].split('|')[2], f'{incorrect_reads_id[i]}-incorrect-{incorrect_start[i]}-{incorrect_end[i]}-{incorrect_reads_kept[incorrect_reads_id[i]]}', len(test_readid_to_read[incorrect_reads_id[i]]), incorrect_strand[i]])
			all_reads_id[incorrect_reads_id[i]] = f'{incorrect_reads_id[i]}-incorrect-{incorrect_start[i]}-{incorrect_end[i]}-{incorrect_reads_kept[incorrect_reads_id[i]]}'

	for i in range(len(correct_reads_id)):
		if correct_reads_id[i] in correct_reads_kept:
			all_reads.append([correct_reads_id[i].split('|')[2], f'{correct_reads_id[i]}-correct-{correct_start[i]}-{correct_end[i]}-{correct_reads_kept[correct_reads_id[i]]}', len(test_readid_to_read[correct_reads_id[i]]), correct_strand[i]])
			all_reads_id[correct_reads_id[i]] = f'{correct_reads_id[i]}-correct-{correct_start[i]}-{correct_end[i]}-{correct_reads_kept[correct_reads_id[i]]}'

	# get contiguous correct and incorrect reads
	cont_reads = []
	cont_reads_id = {}
	for i in range(len(incorrect_reads_id)):
		if incorrect_reads_id[i] in incorrect_reads_kept:
			for j in range(len(correct_reads_id)):
				if correct_reads_id[j] in correct_reads_kept:
					if (correct_start[j] < incorrect_end[i] and correct_end[j] > incorrect_start[i]) or \
						(incorrect_start[i] < correct_end[j] and incorrect_end[i] > correct_start[j]) or \
						(correct_start[j] < incorrect_start[i] and correct_end[j] > incorrect_end[i]) or \
						(incorrect_start[i] < correct_start[j] and incorrect_end[i] > correct_end[j]):
						if correct_strand[j] == 'plus' and incorrect_strand[i] == 'plus':
							if abs(len(test_readid_to_read[incorrect_reads_id[i]])-len(test_readid_to_read[correct_reads_id[j]])) < 200:
								cont_reads.append([correct_reads_id[j].split('|')[2], f'{correct_reads_id[j]}-correct-{correct_start[j]}-{correct_end[j]}-{correct_reads_kept[correct_reads_id[j]]}', len(test_readid_to_read[correct_reads_id[j]]), correct_strand[j], \
									incorrect_reads_id[i].split('|')[2], f'{incorrect_reads_id[i]}-incorrect-{incorrect_start[i]}-{incorrect_end[i]}-{incorrect_reads_kept[incorrect_reads_id[i]]}', len(test_readid_to_read[incorrect_reads_id[i]]), incorrect_strand[i]])
								cont_reads_id[correct_reads_id[j]] = f'{correct_reads_id[j]}-correct-{correct_start[j]}-{correct_end[j]}-{correct_reads_kept[correct_reads_id[j]]}'
								cont_reads_id[incorrect_reads_id[i]] = f'{incorrect_reads_id[i]}-incorrect-{incorrect_start[i]}-{incorrect_end[i]}-{incorrect_reads_kept[incorrect_reads_id[i]]}'

	WriteInputAttentions(os.path.join(args.output_dir, f'{args.testing_genome}_contiguous_reads_kept.tsv'), os.path.join(args.output_dir, f'{args.testing_genome}_contiguous_id.tsv'), cont_reads, cont_reads_id, test_readid_to_read)
	WriteInputAttentions(os.path.join(args.output_dir, f'{args.testing_genome}_all_reads_kept.tsv'), os.path.join(args.output_dir, f'{args.testing_genome}_all_id.tsv'), all_reads, all_reads_id, test_readid_to_read)

def CheckSeqInGene(read_start_pos, read_end_pos, gene_start_pos, gene_end_pos):
	# check if read_id maps to gene
	length_mapped_seq = 0
	if (read_start_pos <= gene_start_pos and read_end_pos >= gene_end_pos) or \
		(read_start_pos <= gene_start_pos and read_end_pos >= gene_start_pos) or \
		(read_start_pos >= gene_start_pos and read_end_pos <= gene_end_pos) or \
		(read_start_pos <= gene_end_pos and read_end_pos >= gene_end_pos):
		if (read_start_pos <= gene_start_pos and read_end_pos >= gene_end_pos):
			length_mapped_seq = read_end_pos - read_start_pos
		elif (read_start_pos <= gene_start_pos and read_end_pos >= gene_start_pos):
			length_mapped_seq = read_end_pos - gene_start_pos
		elif (read_start_pos >= gene_start_pos and read_end_pos <= gene_end_pos):
			length_mapped_seq = read_end_pos - read_start_pos
		elif (read_start_pos <= gene_end_pos and read_end_pos >= gene_end_pos):
			length_mapped_seq = gene_end_pos - read_start_pos
	return length_mapped_seq


def GetGenes(args, annot_info, incorrect_alignments, correct_alignments, sequence_length, readid_to_read, genome_size, incorrect_cs, correct_cs):
	
	correct_genes = defaultdict(list)
	incorrect_genes = defaultdict(list)
	correct_functions = defaultdict(int)
	incorrect_functions = defaultdict(int)
	correct_reads_kept = dict()
	incorrect_reads_kept = dict()
	gene_to_incorrect_reads_kept = dict()
	gene_to_correct_reads_kept = dict()
	scores = {i:0 for i in range(genome_size)}

	outf = open(os.path.join(args.output_dir, 'gene_selection_summary.tsv'), 'w')
	outf.write(f'gene_id\tincorrect_positions\tcorrect_positions\tratio_incorrect\tratio_correct\t'
				f'correct sequences\tincorrect_sequences\n')
	for gene_id, data in annot_info.items():
		gene_start_pos = data[1]
		gene_end_pos = data[2]
		correct_reads = []
		incorrect_reads = []

		correct_mapped_length = dict()
		incorrect_mapped_length = dict()
		for read_id, align_info in incorrect_alignments.items():
			if align_info[1] < align_info[2]:
				read_start_pos = align_info[1]
				read_end_pos = align_info[2]
			else:
				read_start_pos = align_info[2]
				read_end_pos = align_info[1]
			length_mapped_seq = CheckSeqInGene(read_start_pos, read_end_pos, gene_start_pos, gene_end_pos)
			if length_mapped_seq > 0:
				incorrect_reads.append(read_id)
				incorrect_mapped_length[read_id] = length_mapped_seq

		for read_id, align_info in correct_alignments.items():
			if align_info[1] < align_info[2]:
				read_start_pos = align_info[1]
				read_end_pos = align_info[2]
			else:
				read_start_pos = align_info[2]
				read_end_pos = align_info[1]
			length_mapped_seq = CheckSeqInGene(read_start_pos, read_end_pos, gene_start_pos, gene_end_pos)
			if length_mapped_seq > 0:
				correct_reads.append(read_id)

				correct_mapped_length[read_id] = length_mapped_seq

		if len(incorrect_reads) + len(correct_reads) > 0:
			# compare number of correct and incorrect positions mapped to gene
			incorrect_num_pos = sum([incorrect_mapped_length[r] for r in incorrect_reads])
			correct_num_pos = sum([correct_mapped_length[r] for r in correct_reads])

			ratio_incorrect = round(incorrect_num_pos / (correct_num_pos + incorrect_num_pos), 2)
			ratio_correct = round(correct_num_pos / (correct_num_pos + incorrect_num_pos), 2)
			outf.write(f'{gene_id}\t{incorrect_num_pos}\t{correct_num_pos}\t{ratio_incorrect}\t{ratio_correct}\t'
				f'\t{len(correct_mapped_length)}\t{len(incorrect_mapped_length)}\n')
			if ratio_incorrect > 0.5:
				incorrect_genes[gene_id] = [ratio_incorrect, incorrect_num_pos, correct_num_pos, len(incorrect_reads), len(correct_reads), gene_start_pos, gene_end_pos]
				if args.analysis == 'FN':
					for i in range(gene_start_pos, gene_end_pos+1, 1):
						scores[i-1] = ratio_incorrect
				for r in incorrect_reads:
					incorrect_reads_kept[r] = gene_id
				# sel_incorrect_evalue.update(incorrect_evalue)
				# sel_incorrect_pident.update(incorrect_pident)
				if data[0] == 'protein_coding':
					incorrect_functions[data[5]] += 1
				gene_to_incorrect_reads_kept[gene_id] = incorrect_reads

			elif ratio_correct > 0.5:
				correct_genes[gene_id] = [ratio_correct, incorrect_num_pos, correct_num_pos, len(incorrect_reads), len(correct_reads), gene_start_pos, gene_end_pos]
				if args.analysis == 'FP':
					for i in range(gene_start_pos, gene_end_pos+1, 1):
						scores[i-1] = ratio_correct
				for r in correct_reads:
					correct_reads_kept[r] = gene_id
				# sel_correct_evalue.update(correct_evalue)
				# sel_correct_pident.update(correct_pident)
				if data[0] == 'protein_coding':
					correct_functions[data[5]] += 1
				gene_to_correct_reads_kept[gene_id] = correct_reads

	incorrect_functions_sorted = dict(sorted(incorrect_functions.items(), key=lambda item: item[1], reverse=True))
	with open(os.path.join(args.output_dir, f'{args.testing_genome}_incorrect_functions_{args.prob_threshold}.tsv'), 'w') as f:
		for k, v in incorrect_functions_sorted.items():
			f.write(f'{k}\t{v}\n')

	correct_functions_sorted = dict(sorted(correct_functions.items(), key=lambda item: item[1], reverse=True))
	with open(os.path.join(args.output_dir, f'{args.testing_genome}_correct_functions_{args.prob_threshold}.tsv'), 'w') as f:
		for k, v in correct_functions_sorted.items():
			f.write(f'{k}\t{v}\n')

	sel_incorrect_cs = [str(incorrect_cs[r]) for r in list(incorrect_reads_kept.keys())]
	with open(os.path.join(args.output_dir, f'{args.testing_genome}_selected_incorrect_cs_{args.prob_threshold}.tsv'), 'w') as f:
		f.write('\n'.join(sel_incorrect_cs))
	
	sel_correct_cs = [str(correct_cs[r]) for r in list(correct_reads_kept.keys())]
	with open(os.path.join(args.output_dir, f'{args.testing_genome}_selected_correct_cs_{args.prob_threshold}.tsv'), 'w') as f:
		f.write('\n'.join(sel_correct_cs))

	sel_incorrect_length = [str(sequence_length[r]) for r in list(incorrect_reads_kept.keys())]
	with open(os.path.join(args.output_dir, f'{args.testing_genome}_selected_incorrect_length_{args.prob_threshold}.tsv'), 'w') as f:
		f.write('\n'.join(sel_incorrect_length))

	sel_correct_length = [str(sequence_length[r]) for r in list(correct_reads_kept.keys())]
	with open(os.path.join(args.output_dir, f'{args.testing_genome}_selected_correct_length_{args.prob_threshold}.tsv'), 'w') as f:
		f.write('\n'.join(sel_correct_length))


	with open(os.path.join(args.output_dir, f'{args.testing_genome}_correct_genes_{args.prob_threshold}.tsv'), 'w') as f:
		for k, v in correct_genes.items():
			f.write(f'{k}')
			for i in range(len(annot_info[k])):
				f.write(f'\t{annot_info[k][i]}')
			for i in range(len(v)):
				f.write(f'\t{v[i]}')
			f.write('\n')

	with open(os.path.join(args.output_dir, f'{args.testing_genome}_incorrect_genes_{args.prob_threshold}.tsv'), 'w') as f:
		for k, v in incorrect_genes.items():
			f.write(f'{k}')
			for i in range(len(annot_info[k])):
				f.write(f'\t{annot_info[k][i]}')
			for i in range(len(v)):
				f.write(f'\t{v[i]}')
			f.write('\n')

	scores_list = [scores[i] for i in range(genome_size)]
	print(f'incorrect_alignments: {len(incorrect_alignments)}')
	print(f'correct_alignments: {len(correct_alignments)}')
	print(f'correct_reads_kept: {len(correct_reads_kept)}')
	print(f'incorrect_reads_kept: {len(incorrect_reads_kept)}')
	# print(f'sel_correct_evalue: {len(sel_correct_evalue)}')
	# print(f'sel_incorrect_evalue: {len(sel_incorrect_evalue)}')
	# print(f'sel_correct_pident: {len(sel_correct_pident)}')
	# print(f'sel_incorrect_pident: {len(sel_incorrect_pident)}')
	with open(os.path.join(args.output_dir, f'{args.testing_genome}_scores_info.tsv'), 'w') as outf:
		outf.write(f'# incorrect reads kept: {len(incorrect_reads_kept)}\n')
		outf.write(f'# correct reads kept: {len(correct_reads_kept)}\n')
		outf.write(f'incorrect rate all positions:\tmean: {statistics.mean(scores_list)}\tmedian: {statistics.median(scores_list)}\tmin: {min(scores_list)}\tmax: {max(scores_list)}\n')
		# outf.write(f'incorrect evalue:\tmean: {statistics.mean(sel_incorrect_evalue.values())}\tmedian: {statistics.median(sel_incorrect_evalue.values())}\tmin: {min(sel_incorrect_evalue.values())}\tmax: {max(sel_incorrect_evalue.values())}\n')
		# outf.write(f'correct evalue:\tmean: {statistics.mean(sel_correct_evalue.values())}\tmedian: {statistics.median(sel_correct_evalue.values())}\tmin: {min(sel_correct_evalue.values())}\tmax: {max(sel_correct_evalue.values())}\n')
		# outf.write(f'incorrect pident:\tmean: {statistics.mean(sel_incorrect_pident.values())}\tmedian: {statistics.median(sel_incorrect_pident.values())}\tmin: {min(sel_incorrect_pident.values())}\tmax: {max(sel_incorrect_pident.values())}\n')
		# outf.write(f'correct pident:\tmean: {statistics.mean(sel_correct_pident.values())}\tmedian: {statistics.median(sel_correct_pident.values())}\tmin: {min(sel_correct_pident.values())}\tmax: {max(sel_correct_pident.values())}\n')

	# create tsv files with FN and TP reads
	CreateTsvFile(incorrect_reads_kept, readid_to_read, os.path.join(args.output_dir, f'{args.testing_genome}_{args.prob_threshold}_incorrect_reads_genes.tsv'))
	CreateTsvFile(correct_reads_kept, readid_to_read, os.path.join(args.output_dir, f'{args.testing_genome}_{args.prob_threshold}_correct_reads_genes.tsv'))

	return scores_list, incorrect_genes, correct_genes, incorrect_reads_kept, correct_reads_kept, gene_to_incorrect_reads_kept, gene_to_correct_reads_kept


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
		return {}
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
				protein_id = ''
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
					if 'protein_id' in e:
						protein_id = e.split(' ')[2]

				if content[i].rstrip().split('\t')[2] == 'gene':
					genes_type[gene_id] = biotype
					locus_tags_info[gene_id] = [begin, end, old_locus_tag, strand]
				elif content[i].rstrip().split('\t')[2] == 'CDS' and genes_type[gene_id] == 'protein_coding':
					if function == '':
						function = gene
					annot_info[gene_id] = ['protein_coding', begin, end, strand, gene, function, protein_id]
				elif content[i].rstrip().split('\t')[2] == 'transcript' and genes_type[gene_id] == 'tRNA':
					annot_info[gene_id] = ['tRNA', begin, end, strand, gene]
				elif content[i].rstrip().split('\t')[2] == 'transcript' and genes_type[gene_id] == 'rRNA':
					annot_info[gene_id] = ['rRNA', begin, end, strand, gene]
				
				assert gene_id != '', 'gene id should not be unknown'
		
		print(len([k for k, v in annot_info.items() if v[0] == 'protein_coding']))
		print(len([k for k, v in annot_info.items() if v[0] == 'tRNA']))
		print(len([k for k, v in annot_info.items() if v[0] == 'rRNA']))

		return annot_info, locus_tags_info


def GetAlignments(sequences, input_file, sequence_length=None, outfilename=None):
	alignments = defaultdict(list)
	with open(input_file, 'r') as f:
		for line in f:
			seqid = line.rstrip().split(',')[0]
			if seqid in sequences:
				sstart = int(line.rstrip().split(',')[2])
				send = int(line.rstrip().split(',')[3])
				ref_id = line.rstrip().split(',')[1]
				evalue = float(line.rstrip().split(',')[7])
				pident = float(line.rstrip().split(',')[8])
				strand = line.rstrip().split(',')[11]
				if seqid in alignments:
					# get best alignment
					if evalue < alignments[seqid][3] and pident > alignments[seqid][4]:
						alignments[seqid] = [ref_id, sstart, send, evalue, pident, strand]
				else:
					alignments[seqid] = [ref_id, sstart, send, evalue, pident, strand]

	if outfilename:
		with open(outfilename, 'w') as f:
			unmapped_reads_id = list(sequences.difference(set(alignments.keys())))
			if len(unmapped_reads_id) != 0:
				unmapped_reads_length = [sequence_length[r] for r in unmapped_reads_id]
				f.write(f'# unmapped reads:\t{len(unmapped_reads_id)}\nmean reads length:\t{statistics.mean(unmapped_reads_length)}\nmedian reads length:\t{statistics.median(unmapped_reads_length)}\nmin reads length:\t{min(unmapped_reads_length)}\nmax reads length:\t{max(unmapped_reads_length)}')
			else:
				f.write(f'# unmapped reads:\t{len(unmapped_reads_id)}\nmean reads length:\tNA\nmedian reads length:\tNA\nmin reads length:\tNA\nmax reads length:\tNA')

	return alignments


def StoreCS(args, dict_cs, type):
	list_cs = []
	for k, v in dict_cs.items():
		list_cs.append(v)
	with open(os.path.join(args.output_dir, f'{args.testing_genome}_{type}_{args.prob_threshold}.tsv'), 'w') as f:
		f.write('\n'.join([str(x) for x in list_cs]))

def GetSeqLength(args, sequences_id, sequence_length, type):
	if len(sequences_id) != 0:
		seq_length_info = [sequence_length[s] for s in sequences_id]
		print(f'{type}\tmean: {statistics.mean(seq_length_info)}\tmedian: {statistics.median(seq_length_info)}\tmax: {max(seq_length_info)}\tmin: {min(seq_length_info)}')

		with open(os.path.join(args.output_dir, f'{args.testing_genome}_{type}_{args.prob_threshold}_seq_length.tsv'), 'w') as f:
			f.write('\n'.join([str(x) for x in seq_length_info]))


def GetFNTPReads(args, test_ordered_reads_id, test_sequence_length, outfile_sum):
	# get FP and TN reads
	tp_sequences = set()
	fn_sequences = set()
	tp_cs = defaultdict(list)
	fn_cs = defaultdict(list)

	with open(args.testing_results, 'r') as f:
		for count, line in enumerate(f):
			prob = float(line.rstrip().split('\t')[2])
			if prob >= args.prob_threshold:
				if line.rstrip().split('\t')[0] == '1' and line.rstrip().split('\t')[1] == '0':
					fn_sequences.add(test_ordered_reads_id[count])
					fn_cs[test_ordered_reads_id[count]] = prob
				if line.rstrip().split('\t')[0] == '1' and line.rstrip().split('\t')[1] == '1':
					tp_sequences.add(test_ordered_reads_id[count])
					tp_cs[test_ordered_reads_id[count]] = prob

	print(f'#FN for genome {args.testing_genome}: {len(fn_sequences)}')
	print(f'#TP for genome {args.testing_genome}: {len(tp_sequences)}')
	outfile_sum.write(f'{len(fn_sequences)}\t{len(tp_sequences)}\n')
	GetSeqLength(args, list(fn_sequences), test_sequence_length, 'fn')
	GetSeqLength(args, list(tp_sequences), test_sequence_length, 'tp')
	StoreCS(args, fn_cs, 'fn')
	StoreCS(args, tp_cs, 'tp')

	return fn_sequences, tp_sequences, fn_cs, tp_cs


def GetFPTNReads(args, test_ordered_reads_id, test_sequence_length, outfile_sum):
	# get FP and TN reads
	tn_sequences = set()
	fp_sequences = set()
	tn_cs = defaultdict(list)
	fp_cs = defaultdict(list)

	with open(args.testing_results, 'r') as f:
		for count, line in enumerate(f):
			prob = float(line.rstrip().split('\t')[2])
			if prob >= args.prob_threshold:
				if line.rstrip().split('\t')[0] == '0' and line.rstrip().split('\t')[1] == '1':
					fp_sequences.add(test_ordered_reads_id[count])
					fp_cs[test_ordered_reads_id[count]] = prob
				if line.rstrip().split('\t')[0] == '0' and line.rstrip().split('\t')[1] == '0':
					tn_sequences.add(test_ordered_reads_id[count])
					tn_cs[test_ordered_reads_id[count]] = prob


	print(f'#FP for genome {args.testing_genome}: {len(fp_sequences)}')
	print(f'#TN for genome {args.testing_genome}: {len(tn_sequences)}')
	outfile_sum.write(f'{len(fp_sequences)}\t{len(tn_sequences)}\n')
	GetSeqLength(args, list(fp_sequences), test_sequence_length, 'fp')
	GetSeqLength(args, list(tn_sequences), test_sequence_length, 'tn')
	StoreCS(args, fp_cs, 'fp')
	StoreCS(args, tn_cs, 'tn')

	return fp_sequences, tn_sequences, fp_cs, tn_cs


def LoadFnaFile(fasta_file):
	with open(fasta_file, 'r') as f:
		content = f.readlines()
	ordered_reads_id = [content[i].rstrip()[1:] for i in range(0,len(content),2)]
	readsid_to_seq = dict(zip([content[i].rstrip()[1:] for i in range(0, len(content), 2)], [content[i].rstrip() for i in range(1, len(content), 2)]))
	readsid_to_length = dict(zip([content[i].rstrip()[1:] for i in range(0, len(content), 2)], [len(content[i].rstrip()) for i in range(1, len(content), 2)]))

	return readsid_to_seq, readsid_to_length, ordered_reads_id


def LoadTsvFile(tsv_file):
	with open(tsv_file, 'r') as f:
		content = f.readlines()
	ordered_reads_id = [content[i].rstrip().split('\t')[0] for i in range(len(content))]
	readsid_to_seq = dict(zip([content[i].rstrip().split('\t')[0] for i in range(len(content))], [content[i].rstrip().split('\t')[1] for i in range(len(content))]))
	readsid_to_length = dict(zip([content[i].rstrip().split('\t')[0] for i in range(len(content))], [len(content[i].rstrip().split('\t')[1]) for i in range(len(content))]))

	return readsid_to_seq, readsid_to_length, ordered_reads_id

def CheckGenomes(args):
	# load testing fasta file
	with open(args.testing_fasta, "r") as handle:
		test_records = list(SeqIO.parse(handle, "fasta"))

	# load training fasta file
	with open(args.training_fasta, "r") as handle:
		train_records = list(SeqIO.parse(handle, "fasta"))

	assert len(test_records) == 1, f'{args.label}\t{args.testing_genome} has more than 1 chromosome'
	assert len(train_records) == 1, f'{args.label}\t{args.training_genome} has more than 1 chromosome'

	return test_records, train_records
