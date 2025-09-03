import os
import sys
import zipfile
import glob
import math
import json
import argparse
import subprocess
from ete3 import Tree
from collections import defaultdict
import pandas as pd
import multiprocessing as mp
import statistics
import random
from pygenomeviz.parser import Fasta
# from Bio import SeqIO
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
sys.path.append('/work/pi_yingzhang_uri_edu/ccres/tools/DNABERT/examples/data_process_template')
from select_genomes import get_gtdb_info
from process_pretrain_data import sampling, cut_no_overlap
from DL_scripts.create_tfrecords import create_tfrecords
from DL_scripts.tfrecords_utils import *


ncbi_datasets_exec = "/work/pi_yingzhang_uri_edu/ccres/tools/datasets"
anvio_exec_dir = "/work/pi_yingzhang_uri_edu/ccres/conda-envs/anvio-8/bin"
blastp_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/blastp"
blastn_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/blastn"
makeblastdb_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/makeblastdb"


def RunBlastn(output_dir, query, subject, num_processes, outfilename):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # create database
    result = subprocess.run([makeblastdb_exec, '-in', f'{subject}', '-input_type', 'fasta', '-dbtype', 'nucl', '-out', f'{output_dir}/blastdb'])
    # align sequences
    # task = 'megablast'
    task = 'blastn'
    result = subprocess.run([blastn_exec, '-query', f'{query}', '-task', f'{task}', '-db', f'{output_dir}/blastdb', '-out', f'{outfilename}', \
            '-outfmt', "10 delim=, qseqid sseqid sstart send qstart qend qlen evalue pident qseq sseq sstrand", \
            '-max_target_seqs', '5', '-num_threads', f'{num_processes}'])


def GetMatchRegions(args, input_file):
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
			# if pident >= args.min_identity:
            align_coords.append([qstart, qend, pident])

    return align_coords, query_pident


def CalculateANI(args, query_fasta, ref_fasta, output_dir):	
    query_size = Fasta(query_fasta).full_genome_length
    # Align query and reference genomes with blastn 
    RunBlastn(output_dir, query_fasta, ref_fasta, args.num_threads, f'{output_dir}/blastn.out')
    align_coords, query_pident = GetMatchRegions(args, f'{output_dir}/blastn.out')

    # count the number of identical positions across the aligned regions
    # store percentage identity between matching regions
    percent_identity = []
    identical_positions = 0
    for ac in align_coords:
        percent_identity.append(ac[2])
        identical_positions += (ac[2]/100*(ac[1]-ac[0]))
    # get stats on percentage identity
    avg_pct_identity = round(identical_positions/query_size*100,2)
    ani = round(statistics.mean(percent_identity), 2)
    
    return ani, avg_pct_identity, query_size



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


def GetAnnotInfo(args, genome_id, input_dir):

	annot_file = glob.glob(os.path.join(input_dir, 'ncbi_database', f'{genome_id}/ncbi_dataset/data/{genome_id}/genomic.gtf'))
	if len(annot_file) == 0:
		f = open(os.path.join(input_dir, 'Genomes_GTF_missing', f'{genome_id}.txt'), 'w')
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
		
		with open(os.path.join(input_dir, 'ncbi_database', genome_id, f'{genome_id}_genes.tsv'), 'w') as f:
			num_proteins = len([k for k, v in annot_info.items() if v[0] == 'protein_coding'])
			num_rrna = len([k for k, v in annot_info.items() if v[0] == 'rRNA'])
			num_trna = len([k for k, v in annot_info.items() if v[0] == 'tRNA'])
			f.write(f'# genes\t{len(annot_info)}\n# protein coding genes\t{num_proteins}\n# rRNA coding genes\t{num_rrna}\n# tRNA coding genes\t{num_trna}\n')

		return annot_info, locus_tags_info


def RunBlastp(args, genome_id, output_dir, query, num_processes, outfilename, input_dir):
	
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)

	protein_fasta = os.path.join(args.output_dir, 'ncbi_database', genome_id, 'ncbi_dataset', 'data', genome_id, 'protein.faa')
	# create database
	result = subprocess.run([makeblastdb_exec, '-in', f'{protein_fasta}', '-input_type', 'fasta', '-dbtype', 'prot', '-out', f'{output_dir}/blastdb'])
	
	# align amino acid sequences to database or fasta file
	result = subprocess.run([blastp_exec, '-query', f'{query}', '-db', f'{output_dir}/blastdb', '-out', f'{outfilename}', \
	 '-outfmt', "10 delim=, qseqid sseqid sstart send qstart qend qlen evalue pident qseq sseq sstrand", \
	 '-max_target_seqs', '5', '-num_threads', f'{num_processes}'])


def PrepareFasta(genomes):
    # remove plasmids and any genomes with multiple chromosomes  
    genomes_kept = []  
    for g in genomes:
        fasta = glob.glob(os.path.join(args.output_dir, 'ncbi_database', g, 'ncbi_dataset/data', g, '*.fna'))
        if len(fasta) >= 1:
            fasta = fasta[0]
            seq_to_keep = []
            descriptions_to_keep = []
            for record in SeqIO.parse(fasta, "fasta"):
                # remove phages and plasmids
                if 'plasmid' not in record.description and 'Plasmid' not in record.description and 'phage' not in record.description:
                    seq_to_keep.append(str(record.seq))
                    descriptions_to_keep.append(record.description)
            
            if len("".join(seq_to_keep)) >= 500000:
                new_fasta = os.path.join(args.output_dir, 'ncbi_database', g, 'ncbi_dataset/data', g, f'updated_{fasta.split("/")[-1]}')
                # if more than one chromosome, combine chromosomes into one sequence
                new_description = f'{descriptions_to_keep[0]}, combined' if len(descriptions_to_keep) > 1 else descriptions_to_keep[0]
                with open(new_fasta, 'w') as out_fasta:
                    out_fasta.write(f'>{new_description}\n{"".join(seq_to_keep)}\n')
                genomes_kept.append(g)
                
    return genomes_kept

def PrepareContigsDb(args, genome_id):
    # Reformat fasta file
    fasta = glob.glob(os.path.join(args.output_dir, 'ncbi_database', genome_id, 'ncbi_dataset/data', genome_id, 'updated_*.fna'))[0]
    new_fasta = fasta.split('.')[0] + '-fixed.fna'
    result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-script-reformat-fasta'), fasta, '--output-file', new_fasta, '--simplify-names', '--seq-type', 'NT'])
    # Generate contigs databases
    output_db = os.path.join(args.output_dir, 'anvio', f'{genome_id}_out.db')
    result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-gen-contigs-database'), '--contigs-fasta', new_fasta, '--project-name', args.species.replace(" ", "-"), '--output-db-path', output_db])
    # Annotate contigs databases
    result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-run-ncbi-cogs'), '--contigs-db', output_db, '--num-threads', f'{args.num_threads}', '--search-with', 'blastp'])
    result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-run-hmms'), '--contigs-db', output_db, '--num-threads', f'{args.num_threads}'])

def ParseAnvioOutput(args, anvio_output, genomes, gene_category, output_dir, software):
    with open(anvio_output, 'r') as f:
        content = f.readlines()
        id_sequences = [content[i].rstrip()[1:] for i in range(0, len(content), 2)]
        aas_sequences = [content[i].rstrip() for i in range(1, len(content), 2)]
        assert len(aas_sequences) == len(id_sequences)
        # sort sequences based on genome of origin
        genomes_sequences = [id_sequences[i].split('|')[2].split(':')[1] for i in range(len(id_sequences))]
        # correct genomes accession id
        for i in range(len(genomes_sequences)):
            for j in range(len(genomes)):
                if genomes_sequences[i] in genomes[j]:
                    genomes_sequences[i] = genomes[j]

    # create directory to store results
    if not os.path.isdir(os.path.join(output_dir, gene_category, 'blast')):
        os.makedirs(os.path.join(output_dir, gene_category, 'blast'))
    
    outf = open(os.path.join(args.output_dir, f'results_{software}.tsv'), 'a')
    outf_miss = open(os.path.join(args.output_dir, f'problematic_proteins_{software}.tsv'), 'a')
    for genome in genomes:
        if genome in genomes_sequences:
            ids = [id_sequences[i] for i in range(len(id_sequences)) if genomes_sequences[i] == genome]
            sequences = [aas_sequences[i] for i in range(len(aas_sequences)) if genomes_sequences[i] == genome]
            # write sequences to fasta file
            with open(os.path.join(output_dir, gene_category, f'{genome}-anvio-{gene_category}.fna'), 'w') as fna:
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

            # align amino acid sequences to genome
            RunBlastp(args, genome, os.path.join(output_dir, gene_category, 'blast', genome), os.path.join(output_dir, gene_category, f'{genome}-anvio-{gene_category}.fna'), \
                args.num_threads, f'{output_dir}/{gene_category}/blast/{genome}/blastp.out', args.output_dir)

			# parse alignment
            alignments = GetAlignments(ids, f'{output_dir}/{gene_category}/blast/{genome}/blastp.out')
			# get annotations of genome
            annot_info, _ = GetAnnotInfo(args, genome, args.output_dir)
			# get genes id from proteins id
            for seq_id in ids:
                if seq_id in alignments:
                    protein_id = alignments[seq_id][0]
					# get gene id
                    seq_gene_id = 'NA'
                    for gene_id in annot_info.keys():
                        if annot_info[gene_id][-1] == protein_id:
                            seq_gene_id = gene_id
                    if seq_id == "NA":
                        print(f'gene id not found: {genome}\t{protein_id}')
                        sys.exit(1)
                    outf.write(f'{genome}\tprotein\t{seq_gene_id}\t{protein_id}\t{gene_category}\t{annot_info[seq_gene_id][1]}\t{annot_info[seq_gene_id][2]}\t{annot_info[seq_gene_id][5]}\n')
                else:
                    outf_miss.write(f'{genome}\tprotein\t{seq_id}\t{gene_category}\n')
            # add info about non coding genes
            for gene_id, info in annot_info.items():
                if info[0] in ['tRNA','rRNA']:
                    outf.write(f'{genome}\t{info[0]}\t{gene_id}\tNA\tNA\t{info[1]}\t{info[2]}\t{info[4]}\n')
    outf.close()

def RunAnvio(args, genomes):
    # Setup a COG data directory
    result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-setup-ncbi-cogs')])

    # Generate and annotate contigs databases
    with mp.Manager() as manager:
        processes = [mp.Process(target=PrepareContigsDb, args=(args, genomes[i])) for i in range(len(genomes))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    # Create tsv file called genome_storage_input.txt
    with open(os.path.join(args.output_dir, 'anvio', 'genome_storage_input.txt'), 'w') as f:
        f.write("name\tcontigs_db_path\n")
        for genome_id in genomes:
            genome_anvio_db = os.path.join(args.output_dir, 'anvio', f'{genome_id}_out.db')
            f.write(f'{genome_id.split(".")[0]}\t{genome_anvio_db}\n')

    # # Generate a genomes storage
    out_genome_storage = os.path.join(args.output_dir, 'anvio', args.species.replace(" ", "-") + '-GENOMES.db')
    result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-gen-genomes-storage'), '--external-genomes', os.path.join(args.output_dir, 'anvio', 'genome_storage_input.txt'), '--output-file', out_genome_storage])

    # # Run pangenome analysis using NCBI blastp for protein search
    blastp_out = os.path.join(args.output_dir, 'anvio', 'blastp')
    result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-pan-genome'), '--genomes-storage', out_genome_storage, '--project-name', args.species.replace(" ", "-"), '--output-dir', blastp_out, '--num-threads', f'{args.num_threads}', '--use-ncbi-blast', '--mcl-inflation', '10'])

    # # Run pangenome analysis using DIAMOND for protein search
    diamond_out = os.path.join(args.output_dir, 'anvio', 'diamond')
    result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-pan-genome'), '--genomes-storage', out_genome_storage, '--project-name', args.species.replace(" ", "-"), '--output-dir', diamond_out, '--num-threads', f'{args.num_threads}', '--mcl-inflation', '10', '--additional-params-for-seq-search', "--masking 0 --sensitive"])

    # Retrieve singleton gene clusters
    result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-get-sequences-for-gene-clusters'), '--pan-db', os.path.join(blastp_out, args.species.replace(" ", "-") + '-PAN.db'), '--genomes-storage', out_genome_storage, '--max-num-genomes', '1', '--max-num-genes-from-each-genome', '1', '--output-file', os.path.join(blastp_out, 'singleton-gene-clusters.fa')])
    result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-get-sequences-for-gene-clusters'), '--pan-db', os.path.join(diamond_out, args.species.replace(" ", "-") + '-PAN.db'), '--genomes-storage', out_genome_storage, '--max-num-genomes', '1', '--max-num-genes-from-each-genome', '1', '--output-file', os.path.join(diamond_out, 'singleton-gene-clusters.fa')])

    # Retrieve single-copy core genes
    result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-get-sequences-for-gene-clusters'), '--pan-db', os.path.join(blastp_out, args.species.replace(" ", "-") + '-PAN.db'), '--genomes-storage', out_genome_storage, '--min-num-genomes', f'{len(genomes)}', '--min-num-genes-from-each-genome', '1', '--output-file', os.path.join(blastp_out, 'single-copy-core-genes.fa')])
    result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-get-sequences-for-gene-clusters'), '--pan-db', os.path.join(diamond_out, args.species.replace(" ", "-") + '-PAN.db'), '--genomes-storage', out_genome_storage, '--min-num-genomes', f'{len(genomes)}', '--min-num-genes-from-each-genome', '1', '--output-file', os.path.join(diamond_out, 'single-copy-core-genes.fa')])
   
    # Parse anvio output
    ParseAnvioOutput(args, os.path.join(blastp_out, 'single-copy-core-genes.fa'), genomes, 'core', blastp_out, 'blastp')
    ParseAnvioOutput(args, os.path.join(blastp_out, 'singleton-gene-clusters.fa'), genomes, 'accessory', blastp_out, 'blastp')
    ParseAnvioOutput(args, os.path.join(diamond_out, 'single-copy-core-genes.fa'), genomes, 'core', diamond_out, 'diamond')
    ParseAnvioOutput(args, os.path.join(diamond_out, 'singleton-gene-clusters.fa'), genomes, 'accessory', diamond_out, 'diamond')

def GetGenomes(args):
    genomes, ncbi_assembly_level, ncbi_genome_category, ncbi_genome_representation, gtdb_rep_genome, gtdb_taxonomy, ncbi_taxonomy = get_gtdb_info(args.gtdb_info)
    genus = args.species.split(' ')[0]
    genomes_of_interest = {}
    for i in range(len(genomes)):
        if gtdb_taxonomy[i].split(';')[-2].split('__')[1] == args.genus:
            if ncbi_assembly_level[i] == "Complete Genome" and ncbi_genome_category[i] != "derived from metagenome" and ncbi_genome_category[i] != "derived from environmental_sample":
                genomes_of_interest[genomes[i]] = gtdb_taxonomy[i].split(';')[-1].split('__')[1]
    return genomes_of_interest

def GetGenomeAndAnnot(args, genome_id):
    if f'{genome_id}' not in os.listdir(os.path.join(args.output_dir, 'ncbi_database')):
        output_dir = os.path.join(args.output_dir, 'ncbi_database', f'{genome_id}')
        os.makedirs(output_dir)
        os.chdir(output_dir)
		# download feature table in gtf and fasta file
        result = subprocess.run([ncbi_datasets_exec, 'download', 'genome', 'accession', f'{genome_id}', '--include', 'gtf,genome,protein'])
		# unzip output folder
        with zipfile.ZipFile('ncbi_dataset.zip', 'r') as zip_ref:
            zip_ref.extractall(os.getcwd())
        os.chdir(args.output_dir)
    else:
        print(f'{genome_id}\tdownload already done')


def GetAni(args, data, input_file, sp_genome):
    with open(input_file, 'r') as f:
        all_genomes = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[1] for line in f.readlines()}
    
    # # check which fasta files are missing
    # genomes_in_db = os.listdir(os.path.join(args.output_dir, 'ncbi_database'))
    # genomes_to_download = list(set(genomes.values()).difference(genomes_in_db))
    # genomes_downloaded = list(set(genomes.values()).intersection(genomes_in_db))
    # print(f'# genomes downloaded: {len(genomes_downloaded)}')
    # print(f'# genomes to download: {len(genomes_to_download)}')
    # # get fasta files and annotations
    # if len(genomes_to_download) > 0:
        # for genome_id in genomes_to_download:
    for genome_id in all_genomes.values():
        GetGenomeAndAnnot(args, genome_id)
    genomes_kept = PrepareFasta(list(all_genomes.values()))
    genomes = {k:v for k, v in all_genomes.items() if v in genomes_kept}
    print(f'# genomes kept: {len(genomes_kept)}\t{len(genomes)}')
    # get training genomes for negative class
    neg_genomes = list(genomes.values())
    print(f'# negative genomes: {len(neg_genomes)}')
    
    # get gtdb taxonomy info
    list_genomes, _, _, _, _, gtdb_taxonomy, ncbi_taxonomy = get_gtdb_info(args.gtdb_info)

    # compute ani between genomes
    ref_fasta = glob.glob(os.path.join(args.output_dir, 'ncbi_database', sp_genome, 'ncbi_dataset/data', sp_genome, 'updated*.fna'))[0]
    prob_f = open(os.path.join(args.output_dir, 'datasets', data, 'problematic_genomes.tsv'), 'w')
    with open(os.path.join(args.output_dir, 'datasets', data, 'ani.tsv'), 'w') as f:
        for genome_id in neg_genomes:
            # Get fasta file of query genome
            query_fasta = glob.glob(os.path.join(args.output_dir, 'ncbi_database', genome_id, 'ncbi_dataset/data', genome_id, 'updated*.fna'))
            if len(query_fasta) == 1:
                query_fasta = query_fasta[0]
                # get label
                label = ""
                for k, v in genomes.items():
                    if v == genome_id:
                        label = k
                output_dir = os.path.join(args.output_dir, 'datasets', data, 'blast', genome_id)
                ani, avg_pct_identity, query_size = CalculateANI(args, query_fasta, ref_fasta, output_dir)
                # get taxonomy
                if genome_id in list_genomes:
                    idx = list_genomes.index(genome_id)
                    gtdb_tax = gtdb_taxonomy[idx]
                    ncbi_tax = ncbi_taxonomy[idx]
                    f.write(f'{label}\t{genome_id}\t{ani}\t{query_size}\t{gtdb_tax}\t{ncbi_tax}\n')  
                else:
                    if 'GCA' in genome_id:
                        if 'GCF_' + genome_id.split('_')[1] in list_genomes:
                            genome_id = 'GCF_' + genome_id.split('_')[1]
                            idx = list_genomes.index(genome_id)
                            gtdb_tax = gtdb_taxonomy[idx]
                            ncbi_tax = ncbi_taxonomy[idx]
                            f.write(f'{label}\t{genome_id}\t{ani}\t{query_size}\t{gtdb_tax}\t{ncbi_tax}\n')  
                        else:
                            prob_f.write(f'{genome_id}\tgenome id not found in gtdb\n')
                    else:
                        prob_f.write(f'{genome_id}\tgenome id not found in gtdb\n')
            else:
                prob_f.write(f'{genome_id}\tncbi dataset not downloaded\n')
            
def SampleGenome(starts, ends, line):
    seq_length = []
    vector_length = []
    sequences = []
    for i in range(len(starts)):
        new_line = line[starts[i]:ends[i]]
        sentence = get_kmer_sentence(new_line, kmer=1)
        if len(sentence) != 0:
            sequences.append(sentence)
            vector_length.append(len(sentence.split(" ")))
            seq_length.append(len(new_line))
        # else:
        #     print(starts[i], 'empty string')
    # print(min(seq_length), max(seq_length), statistics.mean(seq_length), statistics.median(seq_length))
    # print(min(vector_length), max(vector_length), statistics.mean(vector_length), statistics.median(vector_length))
    return sequences

def CutGenome(cuts, line):
    start = 0
    seq_length = []
    vector_length = []
    sequences = []
    seq_starts = []
    seq_ends = []
    for cut in cuts:
        new_line = line[start:start+cut]
        sentence = get_kmer_sentence(new_line, kmer=1)
        if len(sentence) != 0:
            vector_length.append(len(sentence.split(" ")))
            seq_length.append(len(new_line))
            seq_starts.append(start)
            seq_ends.append(start+cut)
            start += cut
            sequences.append(sentence)
            
    # print(min(seq_length), max(seq_length), statistics.mean(seq_length), statistics.median(seq_length))
    # print(min(vector_length), max(vector_length), statistics.mean(vector_length), statistics.median(vector_length))
    return sequences, seq_starts, seq_ends

def get_kmer_sentence(original_string, kmer=1, stride=1):
    if kmer == -1:
        return original_string

    sentence = ""
    original_string = original_string.replace("\n", "")
    i = 0
    while i < len(original_string)-kmer:
        sentence += original_string[i:i+kmer] + " "
        i += stride
    
    return sentence[:-1].strip("\"")

def GetSequences(args, labels, num, sequences, info):
    for label in labels:
        genome_id = info[label][0]
        fasta = Fasta(glob.glob(os.path.join(args.output_dir, 'ncbi_database', genome_id, 'ncbi_dataset/data', genome_id, 'updated*.fna'))[0])
        starts, ends = sampling(length=int(info[label][2]), kmer=1, sampling_rate=0.5)
        sam_sequences = SampleGenome(starts, ends, fasta.full_genome_seq)
        cuts = cut_no_overlap(length=int(info[label][2]), kmer=1)
        cut_sequences, seq_starts, seq_ends = CutGenome(cuts, fasta.full_genome_seq)
        all_sequences = sam_sequences + cut_sequences
        all_starts = starts + seq_starts
        all_ends = ends + seq_ends
        to_shuffle = list(zip(all_sequences, all_starts, all_ends))
        random.shuffle(to_shuffle)
        sequences[label] = to_shuffle[:num]


def GetGenomeSequence(args, genome, fasta):
    print(f'{genome}\t{fasta.full_genome_length}')
    sam_sequences = []
    sam_starts = []
    sam_ends = []
    for i in range(args.cov):
        run_starts, run_ends = sampling(length=int(fasta.full_genome_length), kmer=1, sampling_rate=0.5)
        run_sequences = SampleGenome(run_starts, run_ends, fasta.full_genome_seq)
        sam_sequences += run_sequences
        sam_starts += run_starts
        sam_ends += run_ends

    cut_sequences = []
    cut_starts = []
    cut_ends = []
    for i in range(args.cov):
        cuts = cut_no_overlap(length=int(fasta.full_genome_length), kmer=1)
        run_sequences, run_starts, run_ends = CutGenome(cuts, fasta.full_genome_seq)
        cut_sequences += run_sequences
        cut_starts += run_starts
        cut_ends += run_ends
    sequences = sam_sequences + cut_sequences
    starts = sam_starts + cut_starts
    ends = sam_ends + cut_ends
    label = [args.label]*len(sequences)
    data = list(zip(sequences, starts, ends, label, [genome]*len(sequences)))
    return data


def CreateTrainValSets(data, all_train_data, all_val_data):
    train_size = round(0.7*len(data))
    val_size = len(data) - train_size
    random.shuffle(data)
    all_train_data += data[:train_size]    
    all_val_data += data[-val_size:]


def PruneTree(args, tree, genomes_to_keep, taxa_2_labels):
    # keep track of the leaves to keep
    leaves_to_keep = set()
    genomes_done = set()
    taxa_ordered = []  # keep track of the taxon associated with the leaves
    for leaf in tree:
        genome_id = leaf.name[3:] if len(leaf.name.split('_')) > 2 else leaf.name
        if genome_id in genomes_to_keep:
            # get taxon of genome
            taxon = genomes_to_keep[genome_id]
            print(taxon)
            if taxon not in leaves_to_keep:
                # update leaf to taxon instead of genome (to only one genome)
                leaf.name = taxon
                leaves_to_keep.add(taxon)
            if genome_id not in genomes_done:
                taxa_ordered.append(taxon)
                genomes_done.add(genome_id)

    print(f'# leaves to keep: {len(leaves_to_keep)}')

    # remove all leaves to not keep
    leaves_to_keep = list(leaves_to_keep)
    # tree.prune(leaves_to_keep, preserve_branch_length=True)
    tree.prune(leaves_to_keep)
    outfile = open(os.path.join(args.output_dir, f'{args.tax_db}_genomes_ordered_{args.rank}_after_pruning'), 'w')
    for leaf in tree:
        outfile.write(f'{leaf.name}\n')

    return taxa_ordered

def CountLeaves(tree):
    num_leaves = 0
    # for leaf in tree:
    for node in tree.traverse():
        if node.is_leaf():
            num_leaves += 1
    print(f'# leaves in tree: {num_leaves}')


def get_node_by_rank(tree, rank, name, gtdb):
    for node in tree.traverse():
        # Get the node's lineage and check if the desired rank is present
        lineage = gtdb.get_lineage(node.taxid)
        names = gtdb.get_taxid_translator(lineage)
        
        # Check if the node is at the specified rank with the correct name
        if node.taxid and names.get(node.taxid) == name and gtdb.get_rank(node.taxid) == rank:
            return node
    return None


def GetTreeDist(args, target_taxa, taxonofinterest, rank):
    rank_to_prefix = {'phylum': 'p__', 'class': 'c__', 'order': 'o__', 'family': 'f__', 'genus': 'g__', 'species': 's__'}
    # load tree
    tree = Tree(args.gtdbtk_tree, quoted_node_names=True, format=1)
    # get nodes associated to target taxa
    taxon_to_node = {}
    for node in tree.traverse("preorder"):
        if node.name and rank_to_prefix[rank] in node.name:
            items = node.name.split(';')
            for item in items:
                if rank_to_prefix[rank] in item:
                    # check if node is at the desired rank
                    if item.split(';').index(node.name) != 0:
                        break
                    taxon = item.split(':')[1].split(';')[0]
                    if taxon in target_taxa:
                        taxon_to_node[taxon] = node
                    
    # measure distances between taxon of interest and the other taxa
    distances = defaultdict(list)
    assert taxonofinterest in taxon_to_node, f'{taxonofinterest} not in gtdb tree!'
    node_toi = taxon_to_node[taxonofinterest]
    for taxon in target_taxa:
        if taxon != taxonofinterest and taxon in taxon_to_node:
            print(taxon, taxon_to_node[taxon].name)
            phylo_distance = node_toi.get_distance(taxon_to_node[taxon], topology_only=False)
            topo_distance = node_toi.get_distance(taxon_to_node[taxon], topology_only=True)
            distances[taxon] = [phylo_distance, topo_distance]
        else:
            distances[taxon] = ['NA', 'NA']
    return distances

    # leaves_to_keep = set()
    # genomes_done = set()
    # for node in tree.traverse("postorder"):
    #     lineage = taxonomy[genome_id]
    #     genome_id = leaf.name[3:] if len(leaf.name.split('_')) > 2 else leaf.name
    #     if genome_id in genomes_to_keep:
    #         # get taxon of genome
    #         taxon = genomes_to_keep[genome_id]
    #         print(taxon)
    #         if taxon not in leaves_to_keep:
    #             # update leaf to taxon instead of genome (to only one genome)
    #             leaf.name = taxon
    #             leaves_to_keep.add(taxon)
    #         if genome_id not in genomes_done:
    #             outfile.write(f'{taxa_2_labels[taxon]}\t{genome_id}\t{taxon}\n')
    #             # print(f'{taxa_2_labels[taxon]}\t{genome_id}\t{taxon}')
    #             genomes_done.add(genome_id)

    # for node in tree.traverse("postorder"):
    #     # get the lineage of the node and the phylum
    #     lineage = gtdb.get_lineage(node.gtdb_taxid)
    #     phylum = None
    #     for taxid in lineage:
    #         rank = gtdb.get_rank(taxid)
    #         if rank == 'phylum':
    #             phylum = gtdb.get_name(taxid)
    #             if phylum in target_taxa:
    #                 break
    #     # verify that every leaf under node has the same phylum
    #     if phylum and not node.is_leaf():
    #         all_descendants_have_same_phylum = True
    #         for leaf in node.iter_leaves():
    #             leaf_lineage = gtdb.get_lineage(leaf.gtdb_taxid)
    #             leaf_phylum = None
    #             for leaf_taxid in leaf_lineage:
    #                 if gtdb.get_rank(leaf_taxid) == 'phylum':
    #                     leaf_phylum = gtdb.get_name(leaf_taxid)
    #                     break
    #             # stop searching if phylum different
    #             if leaf_phylum != phylum:
    #                 all_descendants_have_same_phylum = False
    #                 break
    #     # collapse node and change it's name
    #     if all_descendants_have_same_phylum:
    #         node.delete()
    #         node.name = phylum

def ParseTestingResults(args, genomes, accuracy):
    for genome_id in genomes:
        results_file = os.path.join(args.testing_results, genome_id, 'testing-results.tsv')
        # load sequences
        datafile = os.path.join(args.input_dir, 'datasets', genome_id, 'dataset.tsv')
        with open(datafile, 'r') as f:
            data = f.readlines()
            # # load cog functions
            # cogfile = os.path.join(args.input_dir, 'ncbi_database', genome_id, 'ncbi_dataset/data', genome_id, 'cog_functions.tsv')
            # cog_df = pd.read_csv(cogfile, sep='\t', header=None)
            # cog_df.columns = ['protein_id', 'function']
            # annot_info, _ = GetAnnotInfo(args, genome_id, args.input_dir)
        incorrect = 0
        correct = 0
        probs = []
        with open(results_file, 'r') as f:
            for index, line in enumerate(f, 0):
                true = line.rstrip().split('\t')[0]
                pred = line.rstrip().split('\t')[1]
                prob = float(line.rstrip().split('\t')[2])
        #         start = int(data[index].split('\t')[2])
        #         end = int(data[index].split('\t')[3])
        #         # print(index, line)
        #         # print(start, end)
                if prob >= args.confidence_score:
        #             seq_gene_id = 'NA'
        #             gene_type = 'NA'
        #             protein_id = 'NA'
        #             cog_fn = 'NA'
        #             pangenome = 'NA'
        #             gene = 'NA'
        #             for gene_id, gene_info in annot_info.items():
        #                 start_gene = gene_info[1]
        #                 end_gene = gene_info[2]
        #                 if (start >= start_gene and end <= end_gene) or \
        #                     (start <= start_gene and end >= end_gene) or \
        #                     (start <= start_gene and end >= start_gene) or \
        #                     (start <= end_gene and end >= end_gene):
        #                     seq_gene_id = gene_id
        #                     gene_type = gene_info[0]
        #                     gene = gene_info[4]
                            
        #                     if gene_type == 'protein_coding':
        #                         protein_id = gene_info[-1]
        #                         cog_fn_df = cog_df.loc[cog_df['protein_id'] == protein_id, 'function']
        #                         if len(cog_fn_df) > 0:
        #                             cog_fn = cog_fn_df.iloc[0]
        #                     # print(f'cog function: {cog_fn}')
        #             if genome_id in pan_genomes and seq_gene_id != 'NA':
        #                 # print(f'gene id: {seq_gene_id}')
        #                 # get pangenome info if available
        #                 # pangenome = anvio_df.loc[(anvio_df['genome'] == genome_id) & (anvio_df['gene'] == gene_id), 'pangenome'].tolist()[0]
        #                 if seq_gene_id in d:
        #                     pangenome = d[seq_gene_id][2]
        #                     # print(f'pangenome: {pangenome}')
        #                     # if pangenome == "accessory":
        #                     #     print(f'gene type : {gene_type}')
        #                     #     print(f'gene id: {seq_gene_id}')
        #                     #     print(f'start: {start}')
        #                     #     print(f'end: {end}')
        #                     #     print(cogfile)
        #                     #     print(datafile)
        #             outf_genes.write(f'{label}\t{genome_id}\t{tax}\t{index}\t{seq_gene_id}\t{gene_type}\t{gene}\t{protein_id}\t{cog_fn}\t')
        #             outf_pan.write(f'{label}\t{genome_id}\t{tax}\t{index}\t{seq_gene_id}\t{gene_type}\t{gene}\t{protein_id}\t{cog_fn}\t{pangenome}\t')
        #             classification = 'NA'
                    if true != pred:
                        incorrect += 1
                        classification = 'incorrect'
                    else:
                        classification = 'correct'
                        correct += 1
                    probs.append(prob)
                    # outf_genes.write(f'{classification}\t{prob}\t{start}\t{end}\n')
                    # outf_pan.write(f'{classification}\t{prob}\t{start}\t{end}\n')
        accuracy[genome_id] = round(correct / (correct+incorrect), 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='path to input directory')
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    parser.add_argument('--species', type=str, help='species with GTDB taxonomy', choices=['Prochlorococcus_B marinus_B','Marinobacter psychrophilus','Alteromonas macleodii'])
    parser.add_argument('--genus', type=str, help='genus with GTDB taxonomy', choices=['Prochlorococcus_B','Marinobacter','Alteromonas'])
    parser.add_argument('--gtdb_info', type=str, help='path to GTDB metadata file')
    parser.add_argument('--label', type=str, help='label associated with species')
    parser.add_argument('--min_identity', type=int, help='identity threshold for comparing aligned sequences', default=70)
    parser.add_argument('--cov', type=int, help='coverage for generating sequences', default=1)
    parser.add_argument('--num_threads', type=int, help='number of threads to run anvio pipeline', default=8)
    parser.add_argument('--anvio', action='store_true', default=False, help="perform anvio pangenome analysis")
    parser.add_argument('--datasets', action='store_true', default=False, help="create training and testing datasets", required=('--datadirname' in sys.argv))
    parser.add_argument('--ani', action='store_true', default=False, help="compute ANI between training genome and list of genomes")
    parser.add_argument('--testing_summary', action='store_true', default=False, help="summarize testing results")
    parser.add_argument('--genome_id', type=str, help="genome id used to create testing set")
    parser.add_argument('--train_genome_id', type=str, help="genome id of training genome")
    parser.add_argument('--genomes', type=str, help="file mapping labels to genomes id")
    parser.add_argument('--neg_genome', type=str, help="negative genome to add to dataset")
    parser.add_argument('--datadirname', type=str, help="name appended to output directory")
    parser.add_argument('--anvio_results', type=str, help="path to file called results_blastp.tsv")
    parser.add_argument('--gene_results', type=str, help="path to file results_genes_pan_*.tsv")
    parser.add_argument('--data', type=str, help="type of dataset", choices=['train','test'])
    parser.add_argument('--mapping_file', type=str, help='path to file mapping species labels to rank labels')
    parser.add_argument('--max_read_length', default=250, type=int, help="The length of simulated reads")
    parser.add_argument('--k_value', nargs='+', type=int, help="Size of k-mers")
    parser.add_argument('--masked_lm_prob', default=0.15, type=float, help="Fraction of masked tokens in mlm task")
    parser.add_argument('--step', default=1, type=int, help="Length of step when sliding window over read")
    parser.add_argument('--vocab', type=str, help="Path to directory containing vocabulary files")
    parser.add_argument('--testing_results', type=str, help="path to parent directory containing files with testing results obtained from running DLTODA")
    parser.add_argument('--info_file', type=str, help="path to file containing taxonomic information about genomes")
    parser.add_argument('--confidence_score', type=float, help="confidence threshold for classifications", default=0.0)
    parser.add_argument('--gtdbtk_tree', type=str, help='path to gtdbtk.bac120.classify.tree')

    args = parser.parse_args()
    print(args)
    # create output directory
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if args.anvio:
        if not os.path.isdir(os.path.join(args.output_dir, 'ncbi_database')):
            os.makedirs(os.path.join(args.output_dir, 'ncbi_database'))
        if not os.path.isdir(os.path.join(args.output_dir, 'anvio')):
            os.makedirs(os.path.join(args.output_dir, 'anvio'))
        
        # get genomes from GTDB
        genomes = GetGenomes(args)
        # get fasta files and annotations
        for genome_id in genomes.keys():
            print(genome_id)
            GetGenomeAndAnnot(args, genome_id)
        
        genomes_kept = PrepareFasta(list(genomes.keys()))
        
        # run anvio
        RunAnvio(args, genomes_kept)

    if args.ani:
        if not os.path.isdir(os.path.join(args.output_dir, 'datasets', args.data)):
            os.makedirs(os.path.join(args.output_dir, 'datasets', args.data, 'blast'))

        # compute ani between genomes
        GetAni(args, args.data, args.genomes, args.train_genome_id)

    if args.datasets:
        # # parse results from previous testing round
        # if args.gene_results is not None:
        #     core_protein_gene_count = defaultdict()
        #     acc_protein_gene_count = defaultdict()
        #     rrna_gene_count = defaultdict()
        #     trna_gene_count = defaultdict()
        #     results_df = pd.read_csv(args.gene_results, sep='\t', header=None)

        if not os.path.isdir(os.path.join(args.output_dir, 'datasets', args.data)):
            os.makedirs(os.path.join(args.output_dir, 'datasets', args.data, 'blast'))
            if args.data == 'train':
                os.makedirs(os.path.join(args.output_dir, 'datasets', f'train-{args.datadirname}', 'tfrecords', 'train'))
                os.makedirs(os.path.join(args.output_dir, 'datasets', f'train-{args.datadirname}', 'tfrecords', 'val'))
            else:
                os.makedirs(os.path.join(args.output_dir, 'datasets', f'test-{args.datadirname}', 'tfrecords'))

        # get genomes
        # with open(os.path.join(args.output_dir, 'datasets', args.data, 'ani.tsv'), 'r') as f:
        with open(args.genomes, 'r') as f:
            info = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[1] for line in f.readlines()}
            # info = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[1:] for line in f.readlines()}

        # get sequences for positive class
        # pos_genome = info[args.label][0]
        pos_genome = info[args.label]
        pos_fasta = Fasta(glob.glob(os.path.join(args.output_dir, 'ncbi_database', pos_genome, 'ncbi_dataset/data', pos_genome, 'updated*.fna'))[0])
        data = GetGenomeSequence(args, pos_genome, pos_fasta)
        # pos_sam_starts, pos_sam_ends = sampling(length=int(info[args.label][2]), kmer=1, sampling_rate=0.5)
        # pos_sam_sequences = SampleGenome(pos_sam_starts, pos_sam_ends, pos_fasta.full_genome_seq)
        # cuts = cut_no_overlap(length=int(info[args.label][2]), kmer=1)
        # pos_cut_sequences, pos_cut_starts, pos_cut_ends = CutGenome(cuts, pos_fasta.full_genome_seq)
        # pos_sequences = pos_sam_sequences + pos_cut_sequences
        # pos_starts = pos_sam_starts + pos_cut_starts
        # pos_ends = pos_sam_ends + pos_cut_ends
        # pos_label = [args.label]*len(pos_sequences)
        # data = list(zip(pos_sequences, pos_starts, pos_ends, pos_label, [pos_genome]*len(pos_sequences)))
        
        if args.data == 'train':
             # get sequences for training and validation datasets
            all_train_data = []
            all_val_data = []
            CreateTrainValSets(data, all_train_data, all_val_data)
            print(f'all val: {len(all_val_data)}')
            print(f'all train: {len(all_train_data)}')
        else:
            all_data = data
            print(f'# test sequences: {len(all_data)}')

        # add data from primary negative genome
        if args.neg_genome:
            fasta = Fasta(glob.glob(os.path.join(args.output_dir, 'ncbi_database', args.neg_genome, 'ncbi_dataset/data', args.neg_genome, 'updated*.fna'))[0])
            neg_genome_data = GetGenomeSequence(args, args.neg_genome, fasta)
        print(f'# sequences from negative genome: {len(neg_genome_data)}')
        # obtain sequences from negative class
        # create chunks of genomes
        neg_labels = [l for l, g in info.items() if l != args.label and g != args.neg_genome]
        chunk_size = math.ceil(len(neg_labels)/args.num_threads)
        print(f'# labels per process: {chunk_size}')
        grouped_labels = [neg_labels[i:i+chunk_size] for i in range(0, len(neg_labels), chunk_size)]
        
         # count the number of sequences per negative genome
        num_sequences = len(all_train_data) + len(all_val_data) - len(neg_genome_data) if args.neg_genome != None else len(all_train_data) + len(all_val_data)
        num_seq_per_sp = [num_sequences // len(neg_labels) + (1 if x < num_sequences % len(neg_labels) else 0)  for x in range(len(neg_labels))]
        print(f'{num_sequences}\t{len(neg_labels)}\t{sum(num_seq_per_sp)}\t{len(num_seq_per_sp)}')
        print(num_seq_per_sp)
        with mp.Manager() as manager: # create manager object to allow processes to manipulate python data structures
            sequences = manager.dict()
            # create list of Process objects
            processes = [mp.Process(target=GetSequences, args=(args, grouped_labels[i], num_seq_per_sp[i], sequences, info)) for i in range(len(grouped_labels))]
            for p in processes:
                p.start() # start the processes
            for p in processes:
                p.join() # join the processes, program will hang and wait until all the processes are done
            
            # prepare datasets
            neg_sequences = []
            neg_all_labels = []
            neg_all_genomes = []
            for k, v in sequences.items():
                neg_sequences += v
                neg_all_labels += [k]*len(v)
                neg_all_genomes += [info[k][0]]*len(v)
            neg_all_sequences, neg_all_starts, neg_all_ends = zip(*neg_sequences)
            neg_data = list(zip(neg_all_sequences, neg_all_starts, neg_all_ends, neg_all_labels, neg_all_genomes))

            if args.data == 'train':
                # get sequences for training and validation datasets
                CreateTrainValSets(neg_data+neg_genome_data, all_train_data, all_val_data)
                print(f'all val: {len(all_val_data)}')
                print(f'all train: {len(all_train_data)}')

                # create tsv file with data
                random.shuffle(all_train_data)
                with open(os.path.join(args.output_dir, 'datasets', f'train-{args.datadirname}', 'train_dataset.tsv'), 'w') as f:
                    sequences, starts, ends, labels, genomes  = zip(*all_train_data)
                    for i in range(len(all_train_data)):
                        new_seq = sequences[i].replace(' ', '')
                        f.write(f'{labels[i]}\t{genomes[i]}\t{starts[i]}\t{ends[i]}\t{new_seq}\n')
                random.shuffle(all_val_data)
                with open(os.path.join(args.output_dir, 'datasets', f'train-{args.datadirname}', 'val_dataset.tsv'), 'w') as f:
                    sequences, starts, ends, labels, genomes = zip(*all_val_data)
                    for i in range(len(all_val_data)):
                        new_seq = sequences[i].replace(' ', '')
                        f.write(f'{labels[i]}\t{genomes[i]}\t{starts[i]}\t{ends[i]-1}\t{new_seq}\n')
            else:
                all_data += data
                print(f'# test sequences: {len(all_data)}')
                # create tsv file with data
                with open(os.path.join(args.output_dir, 'datasets', f'test-{args.datadirname}', 'test_dataset.tsv'), 'w') as f:
                    sequences, starts, ends, labels, genomes = zip(*all_data)
                    for i in range(len(all_data)):
                        new_seq = sequences[i].replace(' ', '')
                        f.write(f'{labels[i]}\t{genomes[i]}\t{starts[i]}\t{ends[i]-1}\t{new_seq}\n')

    
        # get dictionary mapping labels to species
        labels_mapping = dict()
        with open(args.mapping_file, 'r') as f:
            for line in f:
                labels_mapping[line.rstrip().split('\t')[0]] = line.rstrip().split('\t')[1]
        
        for k_value in args.k_value:
            kmer_vector_length = args.max_read_length - k_value + 1 if args.step == 1 else args.max_read_length // k_value
            print(f'max read length: {args.max_read_length}\tvector size: {kmer_vector_length}\t{k_value}')
            
            # create tfrecords for bert
            if args.data == 'train':
                output_dir = os.path.join(args.output_dir, 'datasets', 'train', 'tfrecords', 'train', f'{k_value}')
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                input_file = os.path.join(args.output_dir, 'datasets', 'train', 'train_dataset.tsv')
                create_tfrecords(input_file, output_dir, k_value, args.step, args.max_read_length, kmer_vector_length, dict_kmers, labels_mapping, \
                    args.masked_lm_prob, dnabert=True, update_labels=True, bert_step='regular', no_label=False, dataset_type='sim', bert=True)

                output_dir = os.path.join(args.output_dir, 'datasets', 'train', 'tfrecords', 'val', f'{k_value}')
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                input_file = os.path.join(args.output_dir, 'datasets', 'train', 'val_dataset.tsv')
                create_tfrecords(input_file, output_dir, k_value, args.step, args.max_read_length, kmer_vector_length, dict_kmers, labels_mapping, \
                    args.masked_lm_prob, dnabert=True, update_labels=True, bert_step='regular', no_label=False, dataset_type='sim', bert=True)
            else:
                output_dir = os.path.join(args.output_dir, 'datasets', 'test', 'tfrecords', f'{k_value}')
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                    os.makedirs(os.path.join(output_dir, 'bert'))
                    # os.makedirs(os.path.join(output_dir, 'cnn'))
                input_file = os.path.join(args.output_dir, 'datasets', 'test', 'test_dataset.tsv')
                # for bert
                # get dictionary mapping kmers to indexes
                dict_kmers = vocab_dict(f'{args.vocab}/bert/{k_value}mers.txt')
                with open(os.path.join(args.output_dir, 'datasets', 'train', 'tfrecords', 'bert', f'{k_value}-dict.json'), 'w') as f:
                    json.dump(dict_kmers, f)
                create_tfrecords(input_file, os.path.join(output_dir, 'bert'), k_value, args.step, args.max_read_length, kmer_vector_length, dict_kmers, labels_mapping, \
                    args.masked_lm_prob, dnabert=True, update_labels=True, bert_step='regular', no_label=False, dataset_type='sim', bert=True)
                ## for cnn
                # dict_kmers = vocab_dict(f'{args.vocab}/dltoda/{k_value}mers.txt')
                # print(dict_kmers)
                # with open(os.path.join(output_dir, 'cnn', f'{k_value}-dict.json'), 'w') as f:
                #     json.dump(dict_kmers, f)
                # create_tfrecords(input_file, os.path.join(output_dir, 'cnn'), k_value, args.step, args.max_read_length, kmer_vector_length, dict_kmers, labels_mapping, \
                #     args.masked_lm_prob, dnabert=True, update_labels=True, bert_step=None, no_label=False, dataset_type='sim', bert=False)
                
    if args.genome_id is not None:
        fasta = Fasta(glob.glob(os.path.join(args.input_dir, 'ncbi_database', args.genome_id, 'ncbi_dataset/data', args.genome_id, 'updated*.fna'))[0])
        sam_sequences = []
        sam_starts = []
        sam_ends = []
        for i in range(args.cov):
            run_starts, run_ends = sampling(length=int(fasta.full_genome_length), kmer=1, sampling_rate=0.5)
            run_sequences = SampleGenome(run_starts, run_ends, fasta.full_genome_seq)
            sam_sequences += run_sequences
            sam_starts += run_starts
            sam_ends += run_ends

        cut_sequences = []
        cut_starts = []
        cut_ends = []
        for i in range(args.cov):
            cuts = cut_no_overlap(length=int(fasta.full_genome_length), kmer=1)
            run_sequences, run_starts, run_ends = CutGenome(cuts, fasta.full_genome_seq)
            cut_sequences += run_sequences
            cut_starts += run_starts
            cut_ends += run_ends
        all_sequences = sam_sequences + cut_sequences
        all_starts = sam_starts + cut_starts
        all_ends = sam_ends + cut_ends

        output_dir = os.path.join(args.output_dir, 'datasets', args.genome_id)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        # create tsv file with data
        input_file = os.path.join(output_dir, 'dataset.tsv')
        with open(input_file, 'w') as f:
            for i in range(len(all_sequences)):
                new_seq = all_sequences[i].replace(' ', '')
                f.write(f'{args.label}\t{args.genome_id}\t{all_starts[i]}\t{all_ends[i]-1}\t{new_seq}\n')
        
        # get dictionary mapping labels to species
        labels_mapping = dict()
        with open(args.mapping_file, 'r') as f:
            for line in f:
                labels_mapping[line.rstrip().split('\t')[0]] = line.rstrip().split('\t')[1]

        for k_value in args.k_value:
            output_dir = os.path.join(args.output_dir, 'datasets', args.genome_id, 'tfrecords', f'{k_value}')
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
                os.makedirs(os.path.join(output_dir, 'bert'))
                os.makedirs(os.path.join(output_dir, 'cnn'))

            kmer_vector_length = args.max_read_length - k_value + 1 if args.step == 1 else args.max_read_length // k_value
            print(f'max read length: {args.max_read_length}\tvector size: {kmer_vector_length}\t{k_value}')
            
            # for bert
            # get dictionary mapping kmers to indexes
            dict_kmers = vocab_dict(f'{args.vocab}/bert/{k_value}mers.txt')
            with open(os.path.join(output_dir, f'{k_value}-dict.json'), 'w') as f:
                json.dump(dict_kmers, f)
            create_tfrecords(input_file, os.path.join(output_dir, 'bert'), k_value, args.step, args.max_read_length, kmer_vector_length, dict_kmers, labels_mapping, \
                args.masked_lm_prob, dnabert=True, update_labels=True, bert_step='regular', no_label=False, dataset_type='sim', bert=True)
            # for cnn
            # get dictionary mapping kmers to indexes
            dict_kmers = vocab_dict(f'{args.vocab}/dltoda/{k_value}mers.txt')
            print(dict_kmers)
            with open(os.path.join(output_dir, 'cnn', f'{k_value}-dict.json'), 'w') as f:
                json.dump(dict_kmers, f)
            create_tfrecords(input_file, os.path.join(output_dir, 'cnn'), k_value, args.step, args.max_read_length, kmer_vector_length, dict_kmers, labels_mapping, \
                args.masked_lm_prob, dnabert=True, update_labels=True, bert_step=None, no_label=False, dataset_type='sim', bert=False)
                    
    if args.testing_summary:
        # load information about genomes 
        with open(args.info_file, 'r') as f:
            info = {line.rstrip().split('\t')[1]: line.rstrip() for line in f.readlines()} 
        # get testing results
        genomes = [v.split('\t')[1] for v in info.values()]
        chunk_size = math.ceil(len(genomes)/args.num_threads)
        grouped_genomes = [genomes[i:i+chunk_size] for i in range(0, len(genomes), chunk_size)]
        with mp.Manager() as manager:
            accuracy = manager.dict()
            processes = [mp.Process(target=ParseTestingResults, args=(args, grouped_genomes[i], accuracy)) for i in range(len(grouped_genomes))]
            for p in processes:
                p.start()
            for p in processes:
                p.join()

            ranks = {'species': 6, 'genus': 5, 'family': 4, 'order': 3, 'class': 2, 'phylum': 1}
            for rank_name, rank_index in ranks.items():
                print(rank_name)
                taxonofinterest = ''
                for k, v in info.items():
                    if v.split('\t')[0] == args.label:
                        taxonofinterest = v.split('\t')[3].split(';')[rank_index]
                assert len(taxonofinterest) != 0, f'info associated with label {args.label} cannot be found'
                # get distances between taxon of interest and all other tax in the testing set
                target_taxa = {v.split('\t')[3].split(';')[rank_index]: k for k, v in info.items()}
                distances = GetTreeDist(args, list(target_taxa.keys()), taxonofinterest, rank_name)
            # # load info about pangenome analysis
            # anvio_df = pd.read_csv(args.anvio_results, sep='\t', header=None)
            # anvio_df.columns = ['genome','type','gene','protein','pangenome','start','end','function']
            # pan_genomes = set(anvio_df['genome'].tolist())
            # print(anvio_df)
            # print(pan_genomes)

            # d = defaultdict(list)
            # pan_genomes = set()
            # with open(args.anvio_results, 'r') as f:
            #     for line in f:
            #         start_gene = int(line.rstrip().split('\t')[5])
            #         end_gene = int(line.rstrip().split('\t')[6])
            #         gene_type = line.rstrip().split('\t')[4]
            #         gene_id = line.rstrip().split('\t')[2]
            #         d[gene_id] = [start_gene, end_gene, gene_type]
            #         pan_genomes.add(line.rstrip().split('\t')[0])

                # outf = open(os.path.join(args.output_dir, f'results_summary_{rank_name}_{args.confidence_score}.tsv'), 'w')
                # outf_taxa = open(os.path.join(args.output_dir, f'results_taxa_{rank_name}_{args.confidence_score}.tsv'), 'w')
                # outf_genes = open(os.path.join(args.output_dir, f'results_genes_{args.confidence_score}.tsv'), 'w')
                # outf_pan = open(os.path.join(args.output_dir, f'results_genes_pan_{args.confidence_score}.tsv'), 'w')
                
                            
                #     outf.write(f'{label}\t{genome_id}\t{tax}\t{accuracy}\t{correct}\t{incorrect}\t')
                #     outf.write(f'{statistics.median(probs)}\t{statistics.mean(probs)}\t{min(probs)}\t{max(probs)}\n')
                # outf.close()
                # outf_genes.close()
                # outf_pan.close()
                outf = open(os.path.join(args.output_dir, f'results_summary_{rank_name}_{args.confidence_score}.tsv'), 'w')
                for taxon in target_taxa:
                    # get all genomes associated with taxon
                    taxon_genomes = [k for k, v in info.items() if v.split('\t')[3].split(';')[rank_index] == taxon]
                    # get accuracy associated with genomes
                    genomes_accuracy = [accuracy[g] for g in taxon_genomes]
                    outf.write(f'{taxon}\t{statistics.mean(genomes_accuracy)}\t{statistics.median(genomes_accuracy)}\t{min(genomes_accuracy)}\t{max(genomes_accuracy)}\t{statistics.stdev(genomes_accuracy)}\t{distances[taxon][0]}\t{distances[taxon][1]}\n')
                outf.close()






