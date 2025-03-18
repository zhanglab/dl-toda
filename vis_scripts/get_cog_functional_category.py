import sys
import os
import glob
import argparse
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


def RunRPSBLAST(args, protein_id):

	result = subprocess.run([f'{rpsblast_exec}', '-query', f'{args.output_dir}/proteins_fasta/{protein_id}.fna', '-db', '/work/pi_yingzhang_uri_edu/ccres/COG-db/Cog', '-out', f'{args.output_dir}/rpsblast_results/{protein_id}_out.tsv', \
	 '-outfmt', '6', '-num_threads', f'{args.num_processes}'])


def GetCOGFnCat(args, protein_id, cdd_to_cog_df, coglettertofn_dict, cogfncat_dict):
	# get best hit and its CDD ID
	if os.path.getsize(f'{args.output_dir}/rpsblast_results/{protein_id}_out.tsv') != 0:
		with open(f'{args.output_dir}/rpsblast_results/{protein_id}_out.tsv', 'r') as f:
			best_hit = f.readline()
			cdd_id = best_hit.rstrip().split('\t')[1].split(':')[1]

		# get COG ID from CDD ID
		row = cdd_to_cog_df.index[cdd_to_cog_df.iloc[:,0]==np.int64(cdd_id)].tolist()
		assert len(row) == 1, f'CDD ID {cdd_id} has not been found'
		cog_id = cdd_to_cog_df.iloc[row[0],1]
		print('cog id', cog_id)
		# get COG functional letter
		if cog_id not in coglettertofn_dict:
			return 'Function unknown'
		else:
			cog_letter = coglettertofn_dict[cog_id]
			print(cog_letter, cogfncat_dict[cog_letter])
			if len(cog_letter) > 1:
				# retrieve most important function
				cog_letter = cog_letter[0]
			return cogfncat_dict[cog_letter]
	else:
		return 'Function unknown'


def GetSequence(args, protein_id):
	fasta_file = ''
	with open(protein_id_to_faa, 'r') as f:
		for line in f:
			if line.rstrip().split('\t')[0] == protein_id:
				fasta_file = line.rstrip().split('\t')[1]
	assert len(fasta_file) != 0, f'{protein_id} is not in local refseq db'
	fasta = open(os.path.join(args.output_dir, 'proteins_fasta', f'{protein_id}.fna'), 'w')
	with open(os.path.join(refseq_dir, fasta_file)) as handle:
	    for record in SeqIO.parse(handle, "fasta"):
	    	if record.id == protein_id:
	    		print(record.id)
	    		fasta.write(f'>{record.id}\n{record.seq}')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type=str, help='one of the fn_unique_genes, tp_unique_genes, fp_shared_genes, tp_shared_genes files')
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

	# get COG functional category of coding sequences
	outf = open(f'{args.input[:-4]}-w-COG.tsv', 'w')
	print(args.input)
	if '_'.join(args.input.split('/')[-1].split('_')[1:3]) in ['fp_shared', 'tp_unique']:
		index = 18
	else:
		index = 16
	with open(args.input, 'r') as f:
		for line in f:
			if line.rstrip().split('\t')[18] == 'protein_coding':
				protein_id = line.rstrip().split('\t')[-1]
				GetSequence(args, protein_id)
				# get fasta file of protein and run rpsblast to retrieve the associated CDD
				RunRPSBLAST(args, protein_id)
				# get COG functional category
				cog_fn = GetCOGFnCat(args, protein_id, cdd_to_cog_df, coglettertofn_dict, cogfncat_dict)
				print(cog_fn)
				outf.write(line.rstrip() + f'\t{cog_fn}\n') 
			else:
				molecule_type = line.rstrip().split('\t')[16]
				# print(molecule_type)
				outf.write(line.rstrip() + f'\t{molecule_type}\n') 


