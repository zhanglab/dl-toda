import sys
import os
import argparse
import math
import zipfile
import subprocess
import multiprocessing
import random
import statistics
import numpy as np
import json
from collections import defaultdict
from pycirclize import Circos, config
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from pygenomeviz.parser import Fasta
from pygenomeviz.utils import load_example_fasta_dataset, ColorCycler, interpolate_color
from pygenomeviz.align import AlignCoord, Blast
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

#ColorCycler.set_cmap("Set1")

colors_pool = ['blue', 'darkviolet', 'black', 'royalblue', 'darkorange', 'green', 'deeppink', 'red', 'gold', 'grey']

blastn_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/blastn"
makeblastdb_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/makeblastdb"

# QUERY_TRACK_SIZE = 5
MIN_IDENTITY = 70
TICKS_INTERVAL = 500000


def RunBlast(args, output_dir, query, subject=None, db=False, outfilename=None, sam=False):
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
	if db:
		sys.executable = blastn_exec
		process = subprocess.run([sys.executable, '-query', f'{query}', '-db', '/datasets/bio/ncbi-db/2025-01-26/nt', '-out', \
			f'{args.output_dir}/blast/test_fp_blastn.out', '-outfmt', "10 delim=, qseqid sseqid evalue pident sstart send qstart qend length ssciname stitle", \
			'-max_target_seqs', '1', '-num_threads', f'{args.num_processes}'])
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
			 	'-outfmt', "17", '-max_target_seqs', '1', '-num_threads', f'{args.num_processes}'])
		else:
			result = subprocess.run([blastn_exec, '-query', f'{query}', '-db', f'{output_dir}/blastdb', '-out', f'{outfilename}', \
			 '-outfmt', "10 delim=, qseqid sseqid sstart send qstart qend qlen evalue pident qseq sseq sstrand", \
			 '-max_target_seqs', '5', '-num_threads', f'{args.num_processes}'])




def GetGenomesInfo(fasta, type):
	with open(fasta, 'r') as f:
		content = f.readline()
	strain = []
	for e in content.rstrip().split(',')[0].split(' ')[1:]:
		if e not in ['chromosome', 'strain', 'complete', 'genome']:
			print(e)
			strain.append(e)
	if type == 'ref':
		return strain[-1]
	elif type == 'query':
		return ' '.join(strain)


def GetMatchRegions(args, input_file, identity_thr=MIN_IDENTITY):
	align_coords = []
	with open(input_file, 'r') as f:
		for count, line in enumerate(f, 1):
			sstart = int(line.rstrip().split(',')[2])
			send = int(line.rstrip().split(',')[3])
			qstart = int(line.rstrip().split(',')[4])
			qend = int(line.rstrip().split(',')[5])
			pident = float(line.rstrip().split(',')[8])
			qseq = line.rstrip().split(',')[9]
			sseq = line.rstrip().split(',')[10]

			if pident >= identity_thr:
				align_coords.append([qstart, qend, pident])

	return align_coords


def CircosPlot(args, outfigpath):
	# get info on ref genomes
	with open(args.input_ref_file, 'r') as f:
		content = f.readlines()
	ref_names = []
	ref_fasta_files = []
	genomes = []
	for line in content:
		ref_names.append(GetGenomesInfo(line.rstrip().split('\t')[2], 'ref'))
		genomes.append(line.rstrip().split('\t')[1])
		ref_fasta_files.append(line.rstrip().split('\t')[2])

	assert len(colors_pool) >= len(ref_names), 'need more colors'

	# load data from fasta files
	query_fasta = Fasta(args.query_fasta_file) 
	comp_ref_fasta = list(map(Fasta, ref_fasta_files))

	# Initialize circos instance
	circos = Circos(
	    sectors=query_fasta.get_seqid2size(),
		space=0
	)

	query_name = GetGenomesInfo(args.query_fasta_file, 'query')
	circos.text(f'{query_name}\n{query_fasta.full_genome_length:,} bp', size=9, r=22)
	print(f'{query_fasta.full_genome_length:,} bp')

	min_r_pos = 100
	for sector in circos.sectors:
		# Setup outer track
		outer_track = sector.add_track((min_r_pos-0.3, min_r_pos))
		outer_track.axis(fc="black")
		outer_track.xticks_by_interval(TICKS_INTERVAL, label_formatter=lambda v: f"{v/1000000:.1f} Mb",)
		outer_track.xticks_by_interval(100000, tick_length=1, show_label=False)
		min_r_pos -= 1

	genomes_pct_identity = []
	genomes_ani = []
	pct_out = open(os.path.join(args.output_dir, f'pct_identity_matching_regions.tsv'), 'w')
	# Blast genome comparison & plot match blocks
	comp_name2color = {}
	genomes_size = []
	# for idx, ref_fasta in enumerate(comp_ref_fasta):
	for idx, ref_fasta in enumerate(ref_fasta_files):
		genomes_size.append(f'{comp_ref_fasta[idx].full_genome_length:,} bp')
		print(comp_ref_fasta[idx].full_genome_length)
		# store percentage identity between matching regions
		percent_identity = []
		# run blast using pygenomeviz
		align_coords = Blast([query_fasta, comp_ref_fasta[idx]]).run()
		align_coords = AlignCoord.filter(align_coords, identity_thr=MIN_IDENTITY)
		# run blast installed on unity
		# RunBlast(args, os.path.join(args.output_dir, 'blast', genomes[idx]), args.query_fasta_file, subject=[ref_fasta], outfilename=f'{args.output_dir}/blast/{genomes[idx]}/blastn.out')
		# align_coords = GetMatchRegions(args, f'{args.output_dir}/blast/{genomes[idx]}/blastn.out', identity_thr=MIN_IDENTITY)
		# count the number of identical positions across the aligned regions
		identical_positions = 0
		color = colors_pool[idx]
		# comp_name2color[comp_ref_fasta.name] = color
		comp_name2color[genomes[idx]] = color
		aligned_length = []
		for sector in circos.sectors:
			blast_track = sector.add_track((min_r_pos-5, min_r_pos), r_pad_ratio=0.1)
		for ac in align_coords:
			# percent_identity.append(ac[2])
			# identical_positions += (ac[2]/100*(ac[1]-ac[0]))
			# print(ac[2], (ac[1]-ac[0]))
			# rect_color = interpolate_color(color, v=ac[2], vmin=MIN_IDENTITY)
			# blast_track.rect(ac[0], ac[1], color=rect_color)
			print(ac.identity, ac.query_end-ac.query_start)
			percent_identity.append(ac.identity)
			aligned_length.append(ac.query_end-ac.query_start)
			identical_positions += (ac.identity/100*(ac.query_end-ac.query_start))
			blast_track = circos.get_sector(ac.query_name).tracks[-1]
			rect_color = interpolate_color(color, v=ac.identity, vmin=MIN_IDENTITY)
			blast_track.rect(ac.query_start, ac.query_end, color=rect_color)

		min_r_pos -= 5
		# get stats on percentage identity
		avg_pct_identity = round(identical_positions/query_fasta.full_genome_length*100,2)
		pct_out.write(f'{genomes[idx]}\t{ref_names[idx]}\t{identical_positions}\t{avg_pct_identity}%\t{round(statistics.mean(percent_identity), 2)}%\n')
		pct_out.write(f'Percent identity of aligned regions\nmean:{statistics.mean(percent_identity)}\tmedian:{statistics.median(percent_identity)}\tmin:{min(percent_identity)}\tmax:{max(percent_identity)}\n')
		pct_out.write(f'Length of aligned regions\nmean:{statistics.mean(aligned_length)}\tmedian:{statistics.median(aligned_length)}\tmin:{min(aligned_length)}\tmax:{max(aligned_length)}\n')

		genomes_pct_identity.append(avg_pct_identity)
		genomes_ani.append(round(statistics.mean(percent_identity), 2))

	# Save figure
	# Enable annotation text adjustment (Default)
	# config.ann_adjust.enable = True
	fig = circos.plotfig()
	# Add legend
	handles=[Patch(label=f'{ref_names[i]} - {genomes_size[i]} - {genomes_pct_identity[i]}% - {genomes_ani[i]}%', fc=comp_name2color[genomes[i]]) for i in range(len(ref_names))]
	_ = circos.ax.legend(handles=handles, bbox_to_anchor=(0.5, 0.475), loc="center", fontsize=8)
	fig.savefig(outfigpath, dpi=300)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--query_fasta_file', type=str, help='path to query fasta file')
	parser.add_argument('--input_ref_file', type=str, help='path to file containing list of reference fasta files and genomes')
	parser.add_argument('--output_dir', type=str, help='path to output directory')
	parser.add_argument('--num_processes', type=int, help='number of processes to run in parallel')
	args = parser.parse_args()

	if not os.path.isdir(args.output_dir):
		os.makedirs(args.output_dir)

	CircosPlot(args, os.path.join(args.output_dir, f'circos.png'))



