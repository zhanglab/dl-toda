import sys
import os
import glob
import argparse
import datetime
import subprocess
import pandas as pd
import numpy as np
import zipfile
from collections import defaultdict

ncbi_datasets_exec = "/work/pi_yingzhang_uri_edu/ccres/tools/datasets"
rpsblast_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/rpsblast"
cog_db = "/work/pi_yingzhang_uri_edu/ccres/COG-db/Cog"
cddtocog = "/work/pi_yingzhang_uri_edu/ccres/CDD-db/cddid.tbl"
coglettertofn = "/work/pi_yingzhang_uri_edu/ccres/COG2024/cog-24.def.tab"
cogfncat = "/work/pi_yingzhang_uri_edu/ccres/COG2024/cog-24.fun.tab"



def RunRPSBLAST(args):
	fasta_file = os.path.join(args.output_dir, f'protein.faa')
	result = subprocess.run([f'{rpsblast_exec}', '-query', f'{fasta_file}', '-db', '/work/pi_yingzhang_uri_edu/ccres/COG-db/Cog', '-out', f'{args.output_dir}/rpsblast_results/rpsblast_out.tsv', \
	 '-outfmt', '6 delim=, qseqid sseqid evalue pident', '-num_threads', f'{args.num_processes}'])


def GetCOGFnCat(args, cdd_to_cog_df, coglettertofn_dict, cogfncat_dict):

	# get best hit and its CDD ID for each protein query
	proteins_cdd = defaultdict(list)
	if os.path.exists(f'{args.output_dir}/rpsblast_results/rpsblast_out.tsv') and os.path.getsize(f'{args.output_dir}/rpsblast_results/rpsblast_out.tsv') != 0:
		with open(f'{args.output_dir}/rpsblast_results/rpsblast_out.tsv', 'r') as f:
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

	# get COG ID and function for each protein
	proteins_fn = {}
	for count, protein_id in enumerate(proteins_cdd.keys()):
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
					# retrieve most important function
					cog_letter = cog_letter[0]
					proteins_fn[protein_id] = cogfncat_dict[cog_letter]
				else:
					# no letter associated with cog id
					proteins_fn[protein_id] = 'Function unknown'
		else:
			# cdd not found in database
			proteins_fn[protein_id] = 'Function unknown'
	print(len(proteins_cdd), len(proteins_fn))
	with open(os.path.join(args.output_dir, 'cog_functions.tsv'), 'w') as f:
		for k, v in proteins_fn.items():
			f.write(f'{k}\t{v}\n')
	# return proteins_fn

# def GetProteins(args):
# 	if f'{args.genome_id}' not in os.listdir(args.proteins_db):
# 		protein_output_dir = os.path.join(args.proteins_db, f'{args.genome_id}')
# 		os.makedirs(protein_output_dir)
# 		os.chdir(protein_output_dir)
# 		# download feature table in gtf if not present
# 		result = subprocess.run([ncbi_datasets_exec, 'download', 'genome', 'accession', f'{args.genome_id}', '--include', 'protein'])
# 		# unzip output folder
# 		with zipfile.ZipFile('ncbi_dataset.zip', 'r') as zip_ref:
# 			zip_ref.extractall(os.getcwd())
# 		os.chdir(args.input_dir)
# 	else:
# 		print(f'{args.genome_id}\tdownload already done')


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str, help='directory containing ncbi_database directory')
	parser.add_argument('--genome_id', type=str, help='genome accession id')
	parser.add_argument('--proteins_db', type=str, help='path to ncbi protein database')
	parser.add_argument('--num_processes', type=int, help='number of processes to run in parallel')
	args = parser.parse_args()
	
	args.output_dir = os.path.join(args.input_dir, 'ncbi_database', args.genome_id, 'ncbi_dataset/data', args.genome_id)
	# create output directories for rpsblast results
	if not os.path.isdir(os.path.join(args.output_dir, 'rpsblast_results')):
		os.makedirs(os.path.join(args.output_dir, 'rpsblast_results'))

	# # get all input files
	# input_files = glob.glob(os.path.join(args.input_dir, '*_*correct_genes_0.9.tsv'))
	# print(input_files, len(input_files))
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
	
	# get proteins associated with genome
	# GetProteins(args)

	# search proteins against the conserved domain database (CDD) with rpsblast
	RunRPSBLAST(args)

	# get COG function for each protein
	GetCOGFnCat(args, cdd_to_cog_df, coglettertofn_dict, cogfncat_dict)

	# for i in range(len(input_files)):
	# 	print(input_files[i])
	# 	if os.path.exists(f'{input_files[i][:-4]}-w-COG.tsv'):
	# 		os.remove(f'{input_files[i][:-4]}-w-COG.tsv')
	# 	with open(f'{input_files[i][:-4]}-w-COG.tsv', 'w') as outf:
	# 		with open(input_files[i], 'r') as inf:
	# 			for line in inf:
	# 				if line.rstrip().split('\t')[1] == 'protein_coding':
	# 					protein_id = line.rstrip().split('\t')[7]
	# 					if protein_id in proteins_fn:
	# 						outf.write(f'{line.rstrip()}\t{proteins_fn[protein_id]}\n')
	# 					else:
	# 						outf.write(f'{line.rstrip()}\tFunction unknown\n')
	# 				else:
	# 					molecule = line.rstrip().split('\t')[1]
	# 					new_line = '\t'.join(line.rstrip().split('\t')[0:6]) + '\tNA\tNA\t' + '\t'.join(line.rstrip().split('\t')[6:])
	# 					outf.write(f'{new_line}\t{molecule}\n')

			