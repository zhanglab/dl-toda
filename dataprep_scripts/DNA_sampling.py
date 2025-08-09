import os
import sys
import zipfile
import glob
import argparse
import subprocess
from collections import defaultdict
import pandas as pd
import multiprocessing as mp
from Bio import SeqIO
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
sys.path.append('/work/pi_yingzhang_uri_edu/ccres/tools/DNABERT/examples/data_process_template')
from select_genomes import get_gtdb_info
from process_pretrain_data import sampling, cut_no_overlap

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

			if pident >= args.min_identity:
				align_coords.append([qstart, qend, pident])

	return align_coords, query_pident


def CalculateANI(args, query_genome, ref_fasta, output_dir):	
    # Get fasta file of query genome
    query_fasta = glob.glob(os.path.join(args.output_dir, 'ncbi_database', query_genome, 'ncbi_dataset/data', query_genome, '*.fna'))[0]
    
	# Align query and reference genomes with blastn 
    RunBlastn(output_dir, query_fasta, ref_fasta, args.num_processes, f'{output_dir}/blastn.out')
    align_coords, query_pident = GetMatchRegions(args, f'{output_dir}/blastn.out')

	# count the number of identical positions across the aligned regions
    # store percentage identity between matching regions
    percent_identity = []
    for ac in align_coords:
        percent_identity.append(ac[2])
        identical_positions += (ac[2]/100*(ac[1]-ac[0]))
	# get stats on percentage identity
    avg_pct_identity = round(identical_positions/query_fasta.full_genome_length*100,2)
    ani = round(statistics.mean(percent_identity), 2)

    return ani


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

	annot_file = glob.glob(os.path.join(args.output_dir, 'ncbi_database', f'{genome_id}/ncbi_dataset/data/{genome_id}/genomic.gtf'))
	if len(annot_file) == 0:
		f = open(os.path.join(args.output_dir, 'Genomes_GTF_missing', f'{genome_id}.txt'), 'w')
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
		
		with open(os.path.join(args.output_dir, 'ncbi_database', genome_id, f'{genome_id}_genes.tsv'), 'w') as f:
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
        fasta = glob.glob(os.path.join(args.output_dir, 'ncbi_database', g, 'ncbi_dataset/data', g, '*.fna'))[0]
        print(fasta)
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
        print(id_sequences[:10])
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
            print(genome)
            ids = [id_sequences[i] for i in range(len(id_sequences)) if genomes_sequences[i] == genome]
            sequences = [aas_sequences[i] for i in range(len(aas_sequences)) if genomes_sequences[i] == genome]
            print(len(sequences), len(ids))
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
    # # Setup a COG data directory
    # result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-setup-ncbi-cogs')])

    # # Generate and annotate contigs databases
    # with mp.Manager() as manager:
    #     processes = [mp.Process(target=PrepareContigsDb, args=(args, genomes[i])) for i in range(len(genomes))]
    #     for p in processes:
    #         p.start()
    #     for p in processes:
    #         p.join()

    # # Create tsv file called genome_storage_input.txt
    # with open(os.path.join(args.output_dir, 'anvio', 'genome_storage_input.txt'), 'w') as f:
    #     f.write("name\tcontigs_db_path\n")
    #     for genome_id in genomes:
    #         genome_anvio_db = os.path.join(args.output_dir, 'anvio', f'{genome_id}_out.db')
    #         f.write(f'{genome_id.split(".")[0]}\t{genome_anvio_db}\n')

    # # Generate a genomes storage
    # out_genome_storage = os.path.join(args.output_dir, 'anvio', args.species.replace(" ", "-") + '-GENOMES.db')
    # print(out_genome_storage)
    # result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-gen-genomes-storage'), '--external-genomes', os.path.join(args.output_dir, 'anvio', 'genome_storage_input.txt'), '--output-file', out_genome_storage])

    # # Run pangenome analysis using NCBI blastp for protein search
    blastp_out = os.path.join(args.output_dir, 'anvio', 'blastp')
    # result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-pan-genome'), '--genomes-storage', out_genome_storage, '--project-name', args.species.replace(" ", "-"), '--output-dir', blastp_out, '--num-threads', f'{args.num_threads}', '--use-ncbi-blast', '--mcl-inflation', '10'])

    # # Run pangenome analysis using DIAMOND for protein search
    diamond_out = os.path.join(args.output_dir, 'anvio', 'diamond')
    # result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-pan-genome'), '--genomes-storage', out_genome_storage, '--project-name', args.species.replace(" ", "-"), '--output-dir', diamond_out, '--num-threads', f'{args.num_threads}', '--mcl-inflation', '10', '--additional-params-for-seq-search', "--masking 0 --sensitive"])

    # Retrieve singleton gene clusters
    # result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-get-sequences-for-gene-clusters'), '--pan-db', os.path.join(blastp_out, args.species.replace(" ", "-") + '-PAN.db'), '--genomes-storage', out_genome_storage, '--max-num-genomes', '1', '--max-num-genes-from-each-genome', '1', '--output-file', os.path.join(blastp_out, 'singleton-gene-clusters.fa')])
    # result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-get-sequences-for-gene-clusters'), '--pan-db', os.path.join(diamond_out, args.species.replace(" ", "-") + '-PAN.db'), '--genomes-storage', out_genome_storage, '--max-num-genomes', '1', '--max-num-genes-from-each-genome', '1', '--output-file', os.path.join(diamond_out, 'singleton-gene-clusters.fa')])

    # Retrieve single-copy core genes
    # result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-get-sequences-for-gene-clusters'), '--pan-db', os.path.join(blastp_out, args.species.replace(" ", "-") + '-PAN.db'), '--genomes-storage', out_genome_storage, '--min-num-genomes', f'{len(genomes)}', '--min-num-genes-from-each-genome', '1', '--output-file', os.path.join(blastp_out, 'single-copy-core-genes.fa')])
    # result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-get-sequences-for-gene-clusters'), '--pan-db', os.path.join(diamond_out, args.species.replace(" ", "-") + '-PAN.db'), '--genomes-storage', out_genome_storage, '--min-num-genomes', f'{len(genomes)}', '--min-num-genes-from-each-genome', '1', '--output-file', os.path.join(diamond_out, 'single-copy-core-genes.fa')])
   
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
        if gtdb_taxonomy[i].split(';')[-1].split('__')[1] == args.species or gtdb_taxonomy[i].split(';')[-2].split('__')[1] == genus:
            if ncbi_assembly_level[i] == "Complete Genome" and ncbi_genome_category[i] != "derived from metagenome" and ncbi_genome_category[i] != "derived from environmental_sample":
                print(genomes[i])
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


def GetAni(args, data, input_file):
    with open(input_file, 'r') as f:
        genomes = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[1] for line in f.readlines()}
    
    # check which fasta files are missing
    genomes_in_db = os.listdir(os.path.join(args.output_dir, 'ncbi_database'))
    genomes_to_download = list(set(genomes).difference(genomes_downloaded))
    genomes_downloaded = list(set(genomes).intersection(genomes_downloaded))
    
    # get fasta files and annotations
    if len(genomes_to_download) > 0:
        for genome_id in genomes_to_download:
            GetGenomeAndAnnot(args, genome_id)
        genomes_kept = PrepareFasta(genomes_to_download)
        genomes = {k:v for k, v in genomes.items() if v in genomes_downloaded+genomes_kept}

    # get training genomes for positive and negative class
    sp_genome = genomes[args.label]
    neg_genomes = [g for l, g in genomes.items() if l != args.label]
    print(f'# negative genomes: {len(neg_genomes)}')
    
    # get gtdb taxonomy info
    genomes, _, _, _, _, gtdb_taxonomy, ncbi_taxonomy = get_gtdb_info(args.gtdb_info)

    # compute ani between genomes
    ref_fasta = glob.glob(os.path.join(args.output_dir, 'ncbi_database', sp_genome, 'ncbi_dataset/data', sp_genome, '*.fna'))[0]
    with open(os.path.join(args.output_dir, 'datasets', data, 'ani.tsv'), 'w') as f:
        for genome_id in neg_genomes:
            # get label
            for k, v in genomes.items():
                if v == genome_id:
                    f.write(f'{k}\t')
            output_dir = os.path.join(args.output_dir, 'datasets', data, 'blast', genome_id)
            ani = CalculateANI(args, genome_id, ref_fasta, output_dir)
            f.write(f'{genome_id}\t{ani}\t')
            # get taxonomy
            idx = genomes.index(genome_id)
            f.write(f'{gtdb_taxonomy[idx]}\t{ncbi_taxonomy[idx]}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    parser.add_argument('--species', type=str, help='species with GTDB taxonomy', choices=['Prochlorococcus_B marinus_B','Marinobacter psychrophilus','Alteromonas macleodii'])
    parser.add_argument('--gtdb_info', type=str, help='path to GTDB metadata file')
    parser.add_argument('--label', type=str, help='label associated with species')
    parser.add_argument('--min_identity', type=int, help='identity threshold for comparing aligned sequences', default=70)
    parser.add_argument('--num_threads', type=int, help='number of threads to run anvio pipeline', default=8)
    parser.add_argument('--anvio', action='store_true', default=False, help="perform anvio pangenome analysis")
    parser.add_argument('--datasets', action='store_true', default=False, help="create training datasets")
    parser.add_argument('--train_genomes', type=str, help="file mapping labels to training genomes id")
    parser.add_argument('--test_genomes', type=str, help="file mapping labels to testing genomes id")
    args = parser.parse_args()

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

    if args.datasets:
        if not os.path.isdir(os.path.join(args.output_dir, 'datasets')):
            os.makedirs(os.path.join(args.output_dir, 'datasets', 'train', 'blast'))
            os.makedirs(os.path.join(args.output_dir, 'datasets', 'test', 'blast'))
        if not os.path.isdir(os.path.join(args.output_dir, 'ncbi_database')):
            os.makedirs(os.path.join(args.output_dir, 'ncbi_database'))

        # compute ani between genomes
        GetAni(args, 'train', args.train_genomes)
        GetAni(args, 'test', args.test_genomes)
        # prepare training and validation datasets from one training genome
      

    # call dnabert functions (provide the whole genome as input) and return start and end on genome for each sequence



    # shuffle and split sequences between train and val (70/30)

    # write sequences to tsv files + add info (location on chromosome, pangenome info, gene)

    # prepare testing dataset

    # call dnabert script (provide the whole genome as input)

    # write sequences to tsv files + add info (location on chromosome, pangenome info, gene)

    # create tfrecords

    # train model

    # test model