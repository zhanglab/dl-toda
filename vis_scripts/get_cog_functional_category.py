import sys
import os
import glob
import argparse
import subprocess
import pandas as pd
from collections import defaultdict

rpsblast_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/rpsblast"
edirect_exec = "/work/pi_yingzhang_uri_edu/ccres/tools/edirect"
cog_db = "/work/pi_yingzhang_uri_edu/ccres/COG-db/Cog"
cddtocog = "/work/pi_yingzhang_uri_edu/ccres/CDD-db/cddid.tbl"
coglettertofn = "/work/pi_yingzhang_uri_edu/ccres/COG2024/cog-24.def.tab"
cogfncat = "/work/pi_yingzhang_uri_edu/ccres/COG2024/cog-24.fun.tab"


def RunRPSBLAST(args, protein_id):

	# result = subprocess.run([f'{edirect_exec}/esearch', '-query', f'{protein_id}', '-db', 'protein', '>', f'{args.output_dir}/proteins_fasta/esearch_out'], shell=True)
	# result = subprocess.run([f'{edirect_exec}/efetch', '-query', f'{args.output_dir}/proteins_fasta/esearch_out', '-format', 'fasta', '>', f'{args.output_dir}/proteins_fasta/{protein_id}_fna'], shell=True)

	result = subprocess.run(f'{edirect_exec}/esearch -query {protein_id} -db protein > {args.output_dir}/proteins_fasta/esearch_out', shell=True)
	result = subprocess.run([f'{edirect_exec}/efetch -query {args.output_dir}/proteins_fasta/esearch_out -format fasta > {args.output_dir}/proteins_fasta/{protein_id}_fna'], shell=True)

	 # 'protein', '|', f'{edirect_exec}/efetch', '-format', 'fasta', '>', f'{args.output_dir}/proteins_fasta/{protein_id}_fna'], shell=True)

	result = subprocess.run([f'{rpsblast_exec}', '-query', f'{args.output_dir}/proteins_fasta/{protein_id}_fna', '-db', '/work/pi_yingzhang_uri_edu/ccres/COG-db/Cog', '-out', f'{args.output_dir}/rpsblast_results/{protein_id}_out.tsv', \
	 f'{cog_db}', '-outfmt', '6', '-num_threads', f'{args.num_processes}'], shell=True)


def GetCOGFnCat(args, protein_id, cdd_to_cog_df, coglettertofn_dict, cogfncat_dict):
	# get best hit and its CDD ID
	with open(f'{args.output_dir}/rpsblast_results/{protein_id}_out.tsv', 'r') as f:
		best_hit = f.readline()
		cdd_id = best_hit.rstrip().split('\t')[1].split(':')[1]

	# get COG ID from CDD ID
	row = cdd_to_cog_df.index[df.iloc[:,0]==cdd_id].tolist()[0]
	cog_id = cdd_to_cog_df.iloc[row,1]

	# get COG functional letter
	cog_letter = coglettertofn_dict[cog_id]

	return cogfncat_dict[cog_letter]


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
	with open(args.input, 'r') as f:
		for line in f:
			protein_id = line.rstrip().split('\t')[-1]
			# get fasta file of protein and run rpsblast to retrieve the associated CDD
			RunRPSBLAST(args, protein_id)
			# get COG functional category
			cog_fn = GetCOGFnCat(args, protein_id, cdd_to_cog_df, coglettertofn_dict, cogfncat_dict)
			outf.write(line.rstrip() + f'\t{cog_fn}\n') 


