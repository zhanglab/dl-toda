import os
import sys
import zipfile
import glob
import argparse
import subprocess
import multiprocessing as mp
from Bio import SeqIO
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from select_genomes import get_gtdb_info

ncbi_datasets_exec = "/work/pi_yingzhang_uri_edu/ccres/tools/datasets"
anvio_exec_dir = "/work/pi_yingzhang_uri_edu/ccres/conda-envs/anvio-8/bin"


def PrepareFasta(genomes):
    # remove plasmids and any genomes with multiple chromosomes  
    genomes_kept = []  
    for g in genomes:
        fasta = glob.glob(os.path.join(args.output_dir, 'genomes', g, 'ncbi_dataset/data', g, '*.fna'))[0]
        print(fasta)
        seq_to_keep = []
        descriptions_to_keep = []
        for record in SeqIO.parse(fasta, "fasta"):
            # remove phages and plasmids
            if 'plasmid' not in record.description and 'Plasmid' not in record.description and 'phage' not in record.description:
                seq_to_keep.append(str(record.seq))
                descriptions_to_keep.append(record.description)
        
        if len("".join(seq_to_keep)) >= 500000:
            new_fasta = os.path.join(args.output_dir, 'genomes', g, 'ncbi_dataset/data', g, f'updated_{fasta.split("/")[-1]}')
            # if more than one chromosome, combine chromosomes into one sequence
            new_description = f'{descriptions_to_keep[0]}, combined' if len(descriptions_to_keep) > 1 else descriptions_to_keep[0]
            with open(new_fasta, 'w') as out_fasta:
                out_fasta.write(f'>{new_description}\n{"".join(seq_to_keep)}\n')
            genomes_kept.append(g)
                
    return genomes_kept

def PrepareContigsDb(args, genome_id):
    # Reformat fasta file
    fasta = glob.glob(os.path.join(args.output_dir, 'genomes', genome_id, 'ncbi_dataset/data', genome_id, 'updated_*.fna'))[0]
    new_fasta = fasta.split('.')[0] + '-fixed.fna'
    result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-script-reformat-fasta'), fasta, '--output-file', new_fasta, '--simplify-names', '--seq-type', 'NT'])
    # Generate contigs databases
    output_db = os.path.join(args.output_dir, 'anvio', f'{genome_id}_out.db')
    result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-gen-contigs-database'), '--contigs-fasta', new_fasta, '--project-name', args.species.replace(" ", ""), '--output-db-path', output_db])
    # Annotate contigs databases
    result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-run-ncbi-cogs'), '--contigs-db', output_db, '--num_threads', '4', '--search-with', 'blastp'])
    result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-run-hmms'), '--contigs-db', output_db, '--num_threads', '4'])

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

    # # Create tsv file called genome_storage_input.txt
    # with open(os.path.join(args.output_dir, 'anvio', 'genome_storage_input.txt'), 'w') as f:
    #     f.write("name\tcontigs_db_path")
    # while read LINE; do genome_id=$(echo $LINE | rev | cut -f1 -d"/" | rev | cut -f1-2 -d"_" | cut -f1 -d"."); echo -e "$genome_id\t$LINE" >> genome_storage_input.txt; done < anvio-db

#     # Generate a genomes storage
#     anvi-gen-genomes-storage --external-genomes genome_storage_input.txt --output-file label-$LABEL-GENOMES.db

#     # Run pangenome analysis using NCBI blastp for protein search
#     anvi-pan-genome --genomes-storage label-$LABEL-GENOMES.db --project-name label_$(echo $LABEL) --output-dir label_$(echo $LABEL)_blastp_genomes --num-threads 64 --use-ncbi-blast --mcl-inflation 10

#     # Run pangenome analysis using DIAMOND for protein search
#     anvi-pan-genome --genomes-storage label-$LABEL-GENOMES.db --project-name label_$(echo $LABEL) --output-dir label_$(echo $LABEL)_diamond_genomes --num-threads 64 --mcl-inflation 10 --additional-params-for-seq-search "--masking 0 --sensitive"

#     # Retrieve singleton gene clusters
#     anvi-get-sequences-for-gene-clusters --pan-db label_$(echo $LABEL)_blastp_genomes/label_$(echo $LABEL)-PAN.db --genomes-storage label-$(echo $LABEL)-GENOMES.db --max-num-genomes 1 --max-num-genes-from-each-genome 1 --output-file label_$(echo $LABEL)_blastp_genomes/singleton-gene-clusters.fa

def GetGenomes(args):
    genomes, ncbi_assembly_level, ncbi_genome_category, ncbi_genome_representation, gtdb_rep_genome, gtdb_taxonomy, ncbi_taxonomy = get_gtdb_info(args.gtdb_info)
    genus = args.species.split(' ')[0]
    genomes_of_interest = []
    for i in range(len(genomes)):
        if gtdb_taxonomy[i].split(';')[-1].split('__')[1] == args.species or gtdb_taxonomy[i].split(';')[-2].split('__')[1] == genus:
            if ncbi_assembly_level[i] == "Complete Genome" and ncbi_genome_category[i] != "derived from metagenome" and ncbi_genome_category[i] != "derived from environmental_sample":
                print(genomes[i])
                genomes_of_interest.append(genomes[i])
    return genomes_of_interest

def GetGenomeAndAnnot(args, genome_id):
	if f'{genome_id}' not in os.listdir(os.path.join(args.output_dir, 'genomes')):
		output_dir = os.path.join(args.output_dir, 'genomes', f'{genome_id}')
		os.makedirs(output_dir)
		os.chdir(output_dir)
		# download feature table in gtf and fasta file
		result = subprocess.run([ncbi_datasets_exec, 'download', 'genome', 'accession', f'{genome_id}', '--include', 'gtf,genome'])
		# unzip output folder
		with zipfile.ZipFile('ncbi_dataset.zip', 'r') as zip_ref:
			zip_ref.extractall(os.getcwd())
		os.chdir(args.output_dir)
	else:
		print(f'{genome_id}\tdownload already done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    parser.add_argument('--species', type=str, help='species with GTDB taxonomy', choices=['Prochlorococcus_B marinus_B','Marinobacter psychrophilus','Alteromonas macleodii'])
    parser.add_argument('--gtdb_info', type=str, help='path to GTDB metadata file')
    parser.add_argument('--training', action='store_true', default=False, help="train model")
    parser.add_argument('--anvio', action='store_true', default=False, help="perform anvio pangenome analysis")
    parser.add_argument('--train_datasets', action='store_true', default=False, help="create training datasets")
    parser.add_argument('--test_datasets', action='store_true', default=False, help="create testing datasets")
    args = parser.parse_args()

    # create output directory
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    # if args.anvio:
    if not os.path.isdir(os.path.join(args.output_dir, 'genomes')):
        os.makedirs(os.path.join(args.output_dir, 'genomes'))
    if not os.path.isdir(os.path.join(args.output_dir, 'anvio')):
        os.makedirs(os.path.join(args.output_dir, 'anvio'))
    
    # get genomes from GTDB
    genomes = GetGenomes(args)
    # get fasta files and annotations
    for genome_id in genomes:
        print(genome_id)
        GetGenomeAndAnnot(args, genome_id)
    
    genomes_kept = PrepareFasta(genomes)
    
    # run anvio
    # map gene id for all genomes to pangenome info and get stats on pangenome analysis
    RunAnvio(args, genomes_kept)

    # prepare training and validation datasets from one training genome

    # call dnabert script (provide the whole genome as input)

    # shuffle and split sequences between train and val (70/30)

    # prepare testing dataset

    # call dnabert script (provide the whole genome as input)

    # create tfrecords

    # train model

    # test model