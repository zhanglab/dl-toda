import os
import sys
import argparse
import subprocess

blastn_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/blastp"
makeblastdb_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/makeblastdb"

def RunBlast(output_dir, query, num_processes, subject, outfilename):
	
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)

	input_fasta = subject[0]
		
	# create database
	result = subprocess.run([makeblastdb_exec, '-in', f'{input_fasta}', '-input_type', 'fasta', '-dbtype', 'nucl', '-out', f'{output_dir}/blastdb'])
	
	# align amino acid sequences to database or fasta file
	result = subprocess.run([blastn_exec, '-query', f'{query}', '-db', f'{output_dir}/blastdb', '-out', f'{outfilename}', \
	 '-outfmt', "10 delim=, qseqid sseqid sstart send qstart qend qlen evalue pident qseq sseq sstrand", \
	 '-max_target_seqs', '5', '-num_threads', f'{num_processes}'])

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--output_dir', type=str, help='path to output directory')
	parser.add_argument('--num_processes', type=int, help='number of processes to run in parallel')
	parser.add_argument('--anvio_output', type=str, help='parse output files from anvio')
	parser.add_argument('--testing_fasta_files', type=str, help='path to file mapping genomes accession id to path to fasta file')
	args = parser.parse_args()

	anvio_output_type = args.anvio_output.split('/')[-1].split('.')[0]
	print(anvio_output_type)

	if not os.path.isdir(args.output_dir):
		os.makedirs(args.output_dir)
	if not os.path.isdir(os.path.join(args.output_dir, anvio_output_type, 'blast')):
		os.makedirs(os.path.join(args.output_dir, anvio_output_type, 'blast'))

	# create dictionary mapping genome accession id to path of fasta file
	with open(args.testing_fasta_files, 'r') as f:
		content = f.readlines()
		genomes_to_fasta = {content[i].rstrip().split('\t')[0]: content[i].rstrip().split('\t')[1] for i in range(len(content))}

	# parse output of anvio
	with open(args.anvio_output, 'r') as f:
		content = f.readlines()
		id_sequences = [content[i].rstrip() for i in range(0, len(content), 2)]
		aas_sequences = [content[i].rstrip() for i in range(1, len(content), 2)]
		# sort sequences based on genome of origin
		genomes_sequences = [id_sequences[i].split('|')[2].split(':')[1] for i in range(len(id_sequences))]
		assert len(aas_sequences) == len(id_sequences)
		# correct genomes accession id
		for i in range(len(genomes_sequences)):
			for key, value in genomes_to_fasta.items():
				if genomes_sequences[i] in key:
					genomes_sequences[i] = key

	for genome, fasta in genomes_to_fasta.items():
		if genome in genomes_sequences:
			ids = [id_sequences[i] for i in range(len(id_sequences)) if genomes_sequences[i] == genome]
			sequences = [aas_sequences[i] for i in range(len(aas_sequences)) if genomes_sequences[i] == genome]

			# write sequences to fasta file
			with open(os.path.join(args.output_dir, anvio_output_type, f'{genome}-anvio-{anvio_output_type}.fna'), 'w') as outf:
				for i in range(len(ids)):
					outf.write(f'>{ids[i]}\n{sequences[i]}\n')

			# align amino acid sequences to genome
			RunBlast(os.path.join(args.output_dir, anvio_output_type, 'blast', genome), os.path.join(args.output_dir, anvio_output_type, f'{genome}-anvio-{anvio_output_type}.fna'), args.num_processes, \
				fasta, f'{args.output_dir}/{anvio_output_type}/blast/{genome}/blastn.out')



