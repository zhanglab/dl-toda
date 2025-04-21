import os
import sys
import argparse
import subprocess
import zipfile
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from pangenome_utils import GetAlignments, GetAnnotInfo

blastn_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/blastp"
makeblastdb_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/makeblastdb"
ncbi_datasets_exec = "/work/pi_yingzhang_uri_edu/ccres/tools/datasets"

def RunBlast(args, genome_id, output_dir, query, num_processes, outfilename, input_dir):
	
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)

	if f'{genome_id}' not in os.listdir(args.protein_db):
		protein_output_dir = os.path.join(args.protein_db, f'{genome_id}')
		os.makedirs(protein_output_dir)
		os.chdir(protein_output_dir)
		# download feature table in gtf if not present
		result = subprocess.run([ncbi_datasets_exec, 'download', 'genome', 'accession', f'{genome_id}', '--include', 'protein'])
		# unzip output folder
		with zipfile.ZipFile('ncbi_dataset.zip', 'r') as zip_ref:
			zip_ref.extractall(os.getcwd())
		os.chdir(input_dir)
	else:
		print(f'{genome_id}\tdownload already done')

	protein_fasta = os.path.join(args.protein_db, genome_id, 'ncbi_dataset', 'data', genome_id, 'protein.faa')
	# create database
	result = subprocess.run([makeblastdb_exec, '-in', f'{protein_fasta}', '-input_type', 'fasta', '-dbtype', 'prot', '-out', f'{output_dir}/blastdb'])
	
	# align amino acid sequences to database or fasta file
	result = subprocess.run([blastn_exec, '-query', f'{query}', '-db', f'{output_dir}/blastdb', '-out', f'{outfilename}', \
	 '-outfmt', "10 delim=, qseqid sseqid sstart send qstart qend qlen evalue pident qseq sseq sstrand", \
	 '-max_target_seqs', '5', '-num_threads', f'{num_processes}'])


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--output_dir', type=str, help='path to output directory')
	parser.add_argument('--protein_db', type=str, help='path to directory containing protein fasta file associated with genomes')
	parser.add_argument('--num_processes', type=int, help='number of processes to run in parallel')
	parser.add_argument('--anvio_output', type=str, help='parse output files from anvio')
	parser.add_argument('--testing_fasta_files', type=str, help='path to file mapping genomes accession id to path to fasta file')
	parser.add_argument('--annotations_dir', type=str, help='path to directory containing gtf annotations files')
	args = parser.parse_args()

	input_dir = os.getcwd()

	anvio_output_type = args.anvio_output.split('/')[-1].split('.')[0]
	print(anvio_output_type)

	if not os.path.isdir(args.output_dir):
		os.makedirs(args.output_dir)
	if not os.path.isdir(os.path.join(args.output_dir, anvio_output_type, 'blast')):
		os.makedirs(os.path.join(args.output_dir, anvio_output_type, 'blast'))

	# create dictionary mapping genome accession id to path of fasta file
	with open(args.testing_fasta_files, 'r') as f:
		content = f.readlines()
		genomes = [content[i].rstrip().split('\t')[0] for i in range(len(content))]

	# parse output of anvio
	with open(args.anvio_output, 'r') as f:
		content = f.readlines()
		id_sequences = [content[i].rstrip()[1:] for i in range(0, len(content), 2)]
		print(id_sequences[:10])
		aas_sequences = [content[i].rstrip() for i in range(1, len(content), 2)]
		# sort sequences based on genome of origin
		genomes_sequences = [id_sequences[i].split('|')[2].split(':')[1] for i in range(len(id_sequences))]
		assert len(aas_sequences) == len(id_sequences)
		# correct genomes accession id
		for i in range(len(genomes_sequences)):
			for j in range(len(genomes)):
				if genomes_sequences[i] in genomes[j]:
					genomes_sequences[i] = genomes[j]

	outf = open(os.path.join(args.output_dir, f'{anvio_output_type}-genes-id.tsv'), 'w')
	for genome in genomes:
		if genome in genomes_sequences:
			ids = [id_sequences[i] for i in range(len(id_sequences)) if genomes_sequences[i] == genome]
			sequences = [aas_sequences[i] for i in range(len(aas_sequences)) if genomes_sequences[i] == genome]

			# write sequences to fasta file
			with open(os.path.join(args.output_dir, anvio_output_type, f'{genome}-anvio-{anvio_output_type}.fna'), 'w') as fna:
				for i in range(len(ids)):
					fna.write(f'>{ids[i]}\n{sequences[i]}\n')

			# align amino acid sequences to genome
			RunBlast(args, genome, os.path.join(args.output_dir, anvio_output_type, 'blast', genome), os.path.join(args.output_dir, anvio_output_type, f'{genome}-anvio-{anvio_output_type}.fna'), \
				args.num_processes, f'{args.output_dir}/{anvio_output_type}/blast/{genome}/blastp.out', input_dir)

			# parse alignment
			alignments = GetAlignments(ids, f'{args.output_dir}/{anvio_output_type}/blast/{genome}/blastp.out')
			print(alignments)
			# get annotations of genome
			annot_info, _ = GetAnnotInfo(args, genome, input_dir)
			print(annot_info)
			# get genes id from proteins id
			for seq_id in ids:
				if seq_id in alignments:
					protein_id = alignments[seq_id][0]
					# get gene id
					seq_gene_id = 'NA'
					for gene_id in annot_info.keys():
						if annot_info[gene_id][-1] == protein_id:
							seq_gene_id = gene_id
					outf.write(f'{seq_id}\t{protein_id}\t{seq_gene_id}\n')
				else:
					outf.write(f'{seq_id}\tNA\tNA\n')
	outf.close()






