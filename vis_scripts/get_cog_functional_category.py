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
	 '-outfmt', '6', '-num_threads', f'{args.num_processes}'])



# def GetCOGFnCat(args, protein_id, cdd_to_cog_df, coglettertofn_dict, cogfncat_dict, file_num):
# 	# get best hit and its CDD ID
# 	if os.path.exists(f'{args.output_dir}/rpsblast_results/{file_num}_out.tsv') and os.path.getsize(f'{args.output_dir}/rpsblast_results/{protein_id}_out.tsv') != 0:
# 		with open(f'{args.output_dir}/rpsblast_results/{file_num}_out.tsv', 'r') as f:
# 			best_hit = f.readline()
# 			cdd_id = best_hit.rstrip().split('\t')[1].split(':')[1]

# 		# get COG ID from CDD ID
# 		row = cdd_to_cog_df.index[cdd_to_cog_df.iloc[:,0]==np.int64(cdd_id)].tolist()
# 		if len(row) == 1:
# 			cog_id = cdd_to_cog_df.iloc[row[0],1]
# 			# get COG functional letter
# 			if cog_id not in coglettertofn_dict:
# 				return 'Function unknown'
# 			else:
# 				cog_letter = coglettertofn_dict[cog_id]
# 				if len(cog_letter) > 1:
# 					# retrieve most important function
# 					cog_letter = cog_letter[0]
# 				return cogfncat_dict[cog_letter]
# 		else:
# 			# cdd not found in database
# 			return 'Function unknown'
# 	else:
# 		# rpsblast didn't find any hit
# 		return 'Function unknown'
# else:
# 	# protein id not found in local refseq db
# 	return 'Function unknown'


def GetSequence(args, list_proteins_id, file_num, protein_to_faa):
	proteins_missing = []
	with open(os.path.join(args.output_dir, 'proteins_fasta', f'{file_num}_proteins.fna'), 'w') as fasta:
		for protein_id in list_proteins_id:
			if protein_id in protein_to_faa:
				fasta_file = protein_to_faa[protein_id]
			# with open(protein_id_to_faa, 'r') as f:
			# 	for line in f:
			# 		if line.rstrip().split('\t')[0] == protein_id:
			# 			fasta_file = line.rstrip().split('\t')[1]
				print(protein_id, fasta_file)

				with open(os.path.join(refseq_dir, fasta_file)) as handle:
				    for record in SeqIO.parse(handle, "fasta"):
				    	if record.id == protein_id:
				    		fasta.write(f'>{record.id}\n{record.seq}')
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
	input_files = glob.glob(os.path.join(args.input_dir, '**/*unique_genes_*.tsv')) + glob.glob(os.path.join(args.input_dir, '**/*shared_genes_*.tsv'))
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
	
	print('before: loading', datetime.datetime.now())
	protein_to_faa = {}
	with open(protein_id_to_faa, 'r') as f:
		for line in f:
		# while True:
		# 	chunk = f.read(4096)
		# 	print(chunk)
		# 	break
			# for i in range(len(chunk)):
			# 	print(chunk[i])
			protein_to_faa[line.rstrip().split('\t')[0]] = line.rstrip().split('\t')[1]
			# if not chunk:
			# 	break

		# protein_to_faa = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[1] for line in f.readlines()}
	print('after: loading', datetime.datetime.now())
	
	for i in range(len(input_files)):
		# get COG functional category of coding sequences
		print(input_files[i])
		with open(f'{input_files[i][:-4]}-w-COG.tsv', 'w'):
			if '_'.join(input_files[i].split('/')[-1].split('_')[1:3]) in ['fp_shared', 'tp_unique']:
				index = 18
			else:
				index = 16
			print('_'.join(input_files[i].split('/')[-1].split('_')[1:3]), index)
			with open(input_files[i], 'r') as f:
				content = {count: line.rstrip().split('\t') for count, line in enumerate(f.readlines())}
				
				# get sequences of proteins into a fasta file
				list_proteins_id = [line[-1] for line in content.values() if line[index] == 'protein_coding']
				print(len(list_proteins_id))
				proteins_missing = GetSequence(args, list_proteins_id, i, protein_to_faa)
				
				# run rpsblast to get cdd id
				if os.path.exists(os.path.join(args.output_dir, 'proteins_fasta', f'{file_num}_proteins.fna')) and \
					os.path.getsize(os.path.join(args.output_dir, 'proteins_fasta', f'{file_num}_proteins.fna')) != 0:

					RunRPSBLAST(args, os.path.join(args.output_dir, 'proteins_fasta', f'{file_num}_proteins.fna'), i)
				

				# for line in f:
				# 	if line.rstrip().split('\t')[index] == 'protein_coding':
				# 		protein_id = line.rstrip().split('\t')[-1]
				# 		# get COG functional category
				# 		cog_fn = GetCOGFnCat(args, protein_id, cdd_to_cog_df, coglettertofn_dict, cogfncat_dict)
				# 		outf.write(f'{line.rstrip()}\t{cog_fn}\n') 
				# 		print(f'old line: {line}\nnew line: {line.rstrip()}\t{cog_fn}\n')
				# 	else:
				# 		molecule_type = line.rstrip().split('\t')[index]
				# 		# print(molecule_type)
				# 		outf.write(f'{line.rstrip()}\t{molecule_type}\n') 
				# 		print(f'old line: {line}\nnew line: {line.rstrip()}\t{molecule_type}\n')


