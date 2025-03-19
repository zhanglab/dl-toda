import sys
import os
import glob
import argparse
import datetime
import subprocess
import pandas as pd
import numpy as np
from collections import defaultdict
from Bio import SeqIO
import gc

rpsblast_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/rpsblast"
cog_db = "/work/pi_yingzhang_uri_edu/ccres/COG-db/Cog"
cddtocog = "/work/pi_yingzhang_uri_edu/ccres/CDD-db/cddid.tbl"
coglettertofn = "/work/pi_yingzhang_uri_edu/ccres/COG2024/cog-24.def.tab"
cogfncat = "/work/pi_yingzhang_uri_edu/ccres/COG2024/cog-24.fun.tab"
protein_id_to_faa = "/datasets/bio/ncbi-refseq/ftp.ncbi.nih.gov/refseq/release/bacteria/protein_id_to_faa"
refseq_dir = "/datasets/bio/ncbi-refseq/ftp.ncbi.nih.gov/refseq/release/bacteria/"


def RunRPSBLAST(args, fasta_file, file_num):
	result = subprocess.run([f'{rpsblast_exec}', '-query', f'{fasta_file}', '-db', '/work/pi_yingzhang_uri_edu/ccres/COG-db/Cog', '-out', f'{args.output_dir}/rpsblast_results/{file_num}_out.tsv', \
	 '-outfmt', '6 delim=, qseqid sseqid evalue pident', '-num_threads', f'{args.num_processes}'])

def GetCOGFnCat(args, list_proteins_id, cdd_to_cog_df, coglettertofn_dict, cogfncat_dict, file_num):
	proteins_cdd = defaultdict(list)
	# get best hit and its CDD ID
	if os.path.exists(f'{args.output_dir}/rpsblast_results/{file_num}_out.tsv') and os.path.getsize(f'{args.output_dir}/rpsblast_results/{file_num}_out.tsv') != 0:
		with open(f'{args.output_dir}/rpsblast_results/{file_num}_out.tsv', 'r') as f:
			for line in f:
				protein_id = line.rstrip().split('\t')[0]
				cdd_id = line.rstrip().split('\t')[1].split('|')[2]
				evalue = float(line.rstrip().split('\t')[2])
				pident = float(line.rstrip().split('\t')[3])
				
				if protein_id not in proteins_cdd:
					proteins_cdd[protein_id] = [cdd_id, evalue, pident]
				else:
					if evalue < proteins_cdd[protein_id][1] and pident > proteins_cdd[protein_id][2]:
						proteins_cdd[protein_id] = [cdd_id, evalue, pident]

	# get COG ID from CDD ID
	proteins_fn = {}
	for protein_id in list_proteins_id:
		if protein_id in proteins_cdd:
			cdd_id = proteins_cdd[protein_id][0]
			row = cdd_to_cog_df.index[cdd_to_cog_df.iloc[:,0]==np.int64(cdd_id)].tolist()
			if len(row) == 1:
				cog_id = cdd_to_cog_df.iloc[row[0],1]
				# get COG functional letter
				if cog_id not in coglettertofn_dict:
					proteins_fn[protein_id] = 'Function unknown'
				else:
					cog_letter = coglettertofn_dict[cog_id]
					if len(cog_letter) > 0:
						if len(cog_letter) > 1:
							# retrieve most important function
							cog_letter = cog_letter[0]
							proteins_fn[protein_id] = cogfncat_dict[cog_letter]
					else:
						# no letter associated with cog id
						proteins_fn[protein_id] = 'Function unknown'
			else:
				# cdd not found in database
				proteins_fn[protein_id] = 'Function unknown'
		else:
			proteins_fn[protein_id] = 'Function unknown'
		

	return proteins_fn

def GetSequence(args, list_proteins_id, file_num):
	proteins_missing = []
	with open(os.path.join(args.output_dir, 'proteins_fasta', f'{file_num}_proteins.fna'), 'w') as fasta:
		for protein_id in list_proteins_id:
			fasta_file = ''
			with open(protein_id_to_faa, 'r') as f:
				for line in f:
					if line.rstrip().split('\t')[0] == protein_id:
						fasta_file = line.rstrip().split('\t')[1]

			if len(fasta_file) != 0:
				with open(os.path.join(refseq_dir, fasta_file)) as handle:
				    for record in SeqIO.parse(handle, "fasta"):
				    	if record.id == protein_id:
				    		fasta.write(f'>{record.id}\n{record.seq}\n')
			else:
				proteins_missing.append(protein_id)
	return proteins_missing


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str, help='diretory containing results obtained from running false_negatives.py or false_positives.py')
	parser.add_argument('--output_dir', type=str, help='path to output directory')
	parser.add_argument('--num_processes', type=int, help='number of processes to run in parallel')
	args = parser.parse_args()

	# create output directories
	if not os.path.isdir(args.output_dir):
		os.makedirs(args.output_dir)
	if not os.path.isdir(os.path.join(args.output_dir, 'proteins_fasta')):
		os.makedirs(os.path.join(args.output_dir, 'proteins_fasta'))
	if not os.path.isdir(os.path.join(args.output_dir, 'rpsblast_results')):
		os.makedirs(os.path.join(args.output_dir, 'rpsblast_results'))

	# get all input files
	input_files = glob.glob(os.path.join(args.input_dir, '**/*unique_genes_0.9.tsv')) + glob.glob(os.path.join(args.input_dir, '**/*shared_genes_0.9.tsv'))
	print(input_files, len(input_files))
	# load required files
	cdd_to_cog_df = pd.read_csv(cddtocog, sep='\t', header=None)
	
	coglettertofn_dict = defaultdict(str)
	with open(coglettertofn, 'r') as f:
		for line in f:
			coglettertofn_dict[line.rstrip().split('\t')[0]] = line.rstrip().split('\t')[1]

	cogfncat_dict = defaultdict(str)
	with open(cogfncat, 'r') as f:
		for line in f:
			if not line.rstrip().split('\t')[0].isdigit():
				if len(line.rstrip().split('\t')) == 4:
					cogfncat_dict[line.rstrip().split('\t')[0]] = line.rstrip().split('\t')[3]
				else:
					cogfncat_dict[line.rstrip().split('\t')[0]] = line.rstrip().split('\t')[2]
	
	for i in range(len(input_files)):
		# get COG functional category of coding sequences
		print(input_files[i])
		with open(f'{input_files[i][:-4]}-w-COG.tsv', 'w') as outf:
			if '_'.join(input_files[i].split('/')[-1].split('_')[1:3]) in ['fp_shared', 'tp_unique']:
				index = 18
			else:
				index = 16
			print('_'.join(input_files[i].split('/')[-1].split('_')[1:3]), index)
			with open(input_files[i], 'r') as f:
				list_proteins_id = [line.rstrip().split('\t')[-1] for line in f.readlines() if line.rstrip().split('\t')[index] == 'protein_coding']
				
			# get sequences of proteins into a fasta file
			print(len(list_proteins_id))
			proteins_missing = GetSequence(args, list_proteins_id, i)
			print(f'# proteins missing: {len(proteins_missing)}')
			# run rpsblast to get cdd id
			if os.path.exists(os.path.join(args.output_dir, 'proteins_fasta', f'{i}_proteins.fna')) and \
				os.path.getsize(os.path.join(args.output_dir, 'proteins_fasta', f'{i}_proteins.fna')) != 0:

				RunRPSBLAST(args, os.path.join(args.output_dir, 'proteins_fasta', f'{i}_proteins.fna'), i)

			proteins_fn = GetCOGFnCat(args, list_proteins_id, cdd_to_cog_df, coglettertofn_dict, cogfncat_dict, i)
			with open(input_files[i], 'r') as f:
				for line in f:
					if line.rstrip().split('\t')[index] == 'protein_coding':
						protein_id = line.rstrip().split('\t')[-1]
						if protein_id in proteins_fn:
							outf.write(f'{line.rstrip()}\t{proteins_fn[protein_id]}\n')
					else:
						molecule = line.rstrip().split('\t')[index]
						outf.write(f'{line.rstrip()}\t{molecule}\n')

			