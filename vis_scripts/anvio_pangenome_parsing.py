import os
import sys
import argparse
import subprocess
from collections import defaultdict
import zipfile
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from pangenome_utils import GetAlignments, GetAnnotInfo, CheckSeqInGene

blastn_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/blastn"
makeblastdb_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/makeblastdb"
ncbi_datasets_exec = "/work/pi_yingzhang_uri_edu/ccres/tools/datasets"

def RunBlast(args, input_fasta, output_dir, query, num_processes, outfilename):
	
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)

	# if f'{genome_id}' not in os.listdir(args.protein_db):
	# 	protein_output_dir = os.path.join(args.protein_db, f'{genome_id}')
	# 	os.makedirs(protein_output_dir)
	# 	os.chdir(protein_output_dir)
	# 	# download feature table in gtf if not present
	# 	result = subprocess.run([ncbi_datasets_exec, 'download', 'genome', 'accession', f'{genome_id}', '--include', 'protein'])
	# 	# unzip output folder
	# 	with zipfile.ZipFile('ncbi_dataset.zip', 'r') as zip_ref:
	# 		zip_ref.extractall(os.getcwd())
	# 	os.chdir(input_dir)
	# else:
	# 	print(f'{genome_id}\tdownload already done')

	# protein_fasta = os.path.join(args.protein_db, genome_id, 'ncbi_dataset', 'data', genome_id, 'protein.faa')
	# create database
	# result = subprocess.run([makeblastdb_exec, '-in', f'{protein_fasta}', '-input_type', 'fasta', '-dbtype', 'prot', '-out', f'{output_dir}/blastdb'])
	
	# align amino acid sequences to database or fasta file
	# result = subprocess.run([blastn_exec, '-query', f'{query}', '-db', f'{output_dir}/blastdb', '-out', f'{outfilename}', \
	#  '-outfmt', "10 delim=, qseqid sseqid sstart send qstart qend qlen evalue pident qseq sseq sstrand", \
	#  '-max_target_seqs', '5', '-num_threads', f'{num_processes}'])

	# create database
	result = subprocess.run([makeblastdb_exec, '-in', f'{input_fasta}', '-input_type', 'fasta', '-dbtype', 'nucl', '-out', f'{output_dir}/blastdb'])
		

	# align dna sequence to genome
	result = subprocess.run([blastn_exec, '-task', 'blastn', '-query', f'{query}', '-db', f'{output_dir}/blastdb', '-out', f'{outfilename}', \
			 '-outfmt', "10 delim=, qseqid sseqid sstart send qstart qend qlen evalue pident qseq sseq sstrand", \
			 '-num_threads', f'{num_processes}'])


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

	if anvio_output_type == 'single-copy-core-genes':
		gene_category = 'core'
	elif anvio_output_type == 'singleton-gene-clusters':
		gene_category = 'accessory'

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
		id_sequences = [content[i].rstrip()[1:] for i in range(0, len(content), 2)]
		dna_sequences = [content[i].rstrip() for i in range(1, len(content), 2)]
		# sort sequences based on genome of origin
		genomes_sequences = [id_sequences[i].split('|')[2].split(':')[1] for i in range(len(id_sequences))]
		assert len(dna_sequences) == len(id_sequences)
		# correct genomes accession id
		for i in range(len(genomes_sequences)):
			for genome in genomes_to_fasta.keys():
				if genomes_sequences[i] in genome:
					genomes_sequences[i] = genome

	outf = open(os.path.join(args.output_dir, f'{anvio_output_type}-genes-id.tsv'), 'w')
	for genome in genomes_to_fasta.keys():
		print(genome)
		# retrieve dna sequences and sequences id
		if genome in genomes_sequences:
			ids = [id_sequences[i] for i in range(len(id_sequences)) if genomes_sequences[i] == genome]
			sequences = [dna_sequences[i] for i in range(len(dna_sequences)) if genomes_sequences[i] == genome]

			# write sequences to fasta file
			with open(os.path.join(args.output_dir, anvio_output_type, f'{genome}-anvio-{anvio_output_type}.fna'), 'w') as fna:
				for i in range(len(ids)):
					# remove any - from sequence
					if '-' in sequences[i]:
						updated_sequence = ''
						for j in range(len(sequences[i])):
							if sequences[i][j] != '-':
								updated_sequence += sequences[i][j]
					else:
						updated_sequence = sequences[i]
					fna.write(f'>{ids[i]}\n{updated_sequence}\n')

			# align dna sequences to genome
			RunBlast(args, genomes_to_fasta[genome], os.path.join(args.output_dir, anvio_output_type, 'blast', genome), os.path.join(args.output_dir, anvio_output_type, f'{genome}-anvio-{anvio_output_type}.fna'), \
				args.num_processes, f'{args.output_dir}/{anvio_output_type}/blast/{genome}/blastn.out')

			# parse alignment
			alignments = GetAlignments(ids, f'{args.output_dir}/{anvio_output_type}/blast/{genome}/blastn.out')
			print('# sequences', len(alignments))
			# get annotations of genome
			annot_info, _ = GetAnnotInfo(args, genome, input_dir)
			print('# genes', len(annot_info))
			# get genes id from annotations
			seq_in_genes = defaultdict(list)
			for seq_id, align_info in alignments.items():
				if align_info[1] < align_info[2]:
					seq_start_pos = align_info[1]
					seq_end_pos = align_info[2]
				else:
					seq_start_pos = align_info[2]
					seq_end_pos = align_info[1]

				for gene_id, data in annot_info.items():
					gene_start_pos = data[1]
					gene_end_pos = data[2]
				
					length_mapped_seq = CheckSeqInGene(seq_start_pos, seq_end_pos, gene_start_pos, gene_end_pos)
					if length_mapped_seq != 0:
						seq_in_genes[seq_id].append(gene_id)
			
			for key, value in seq_in_genes.items():
				if len(value) > 1:
					print(key, value)
					break
			break


				# outf.write()

	# 		for seq_id in ids:
	# 			if seq_id in alignments:
	# 				protein_id = alignments[seq_id][0]
	# 				# get gene id
	# 				seq_gene_id = 'NA'
	# 				for gene_id in annot_info.keys():
	# 					if annot_info[gene_id][-1] == protein_id:
	# 						seq_gene_id = gene_id
	# 				outf.write(f'{seq_id}\t{protein_id}\t{seq_gene_id}\t{gene_category}\n')
	# 			else:
	# 				outf.write(f'{seq_id}\tNA\tNA\t{gene_category}\n')
	# outf.close()






