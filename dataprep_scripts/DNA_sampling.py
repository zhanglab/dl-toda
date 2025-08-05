import os
import sys
import zipfile
import glob
import argparse
import subprocess
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from select_genomes import get_gtdb_info

ncbi_datasets_exec = "/work/pi_yingzhang_uri_edu/ccres/tools/datasets"
anvio_exec_dir = "/work/pi_yingzhang_uri_edu/ccres/conda-envs/anvio-8/bin"


def PrepareFasta(genomes):
    # remove plasmids and any genomes with multiple chromosomes    
    for g in genomes:
        fasta = glob.glob(os.path.join(os.getcwd(), 'genomes', g, 'ncbi_dataset/data', g, '*.fna'))[0]
        with open(fasta, 'r') as f:
            for line in f:
                if line[0] == '>':
                    print(line[0].lower())
                    if 'plasmid' in line[0].lower():
                        print('yes')
        break
    # 


# def RunAnvio(args):
    

#     # Reformat fasta files
#     result = subprocess.run([os.path.join(anvio_exec_dir, 'anvi-script-reformat-fasta'), 'download', 'genome', 'accession', f'{genome_id}', '--include', 'gtf,genome'])
    
#     cat fasta.tsv | parallel -j 8 --colsep '\t' anvi-script-reformat-fasta {2} --output-file {1}-fixed.fna --simplify-names --seq-type NT

#     # Fix format of fasta files header
#     find ~+ -name "*-fixed.fna" > new-fasta.tsv
#     while read LINE; do genome=$(echo $LINE | rev | cut -f1 -d"/" | rev | cut -f1 -d"-"); echo -e "$genome\t$LINE" >> fixed-fasta.tsv; done < new-fasta.tsv

#     # Generate contigs databases
#     cat fixed-fasta.tsv | parallel --colsep '\t' -j 8 anvi-gen-contigs-database --contigs-fasta {2} --project-name label_239 --output-db-path {1}_out.db

#     # Setup a COG data directory
#     anvi-setup-ncbi-cogs

#     # Annotate contigs databases
#     find ~+ -name "*_out.db" > anvio-db
#     cat anvio-db | parallel -j 2 anvi-run-ncbi-cogs --contigs-db {} --num-threads 32 --search-with blastp
#     cat anvio-db | parallel -j 2 anvi-run-hmms --contigs-db {} --num-threads 32

#     # Create tsv file called genome_storage_input.txt
#     echo -e "name\tcontigs_db_path" > genome_storage_input.txt
#     while read LINE; do genome_id=$(echo $LINE | rev | cut -f1 -d"/" | rev | cut -f1-2 -d"_" | cut -f1 -d"."); echo -e "$genome_id\t$LINE" >> genome_storage_input.txt; done < anvio-db

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
    args = parser.parse_args()

    # create output directory
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.isdir(os.path.join(args.output_dir, 'genomes')):
        os.makedirs(os.path.join(args.output_dir, 'genomes'))
    
    # get genomes from GTDB
    genomes = GetGenomes(args)
    # get fasta files and annotations
    for genome_id in genomes:
        print(genome_id)
        GetGenomeAndAnnot(args, genome_id)
    
    # run anvio
    # map gene id for all genomes to pangenome info and get stats on pangenome analysis

    # prepare training and validation datasets from one training genome

    # call dnabert script (provide the whole genome as input)

    # shuffle and split sequences between train and val (70/30)

    # prepare testing dataset

    # call dnabert script (provide the whole genome as input)

    # create tfrecords

    # train model

    # test model