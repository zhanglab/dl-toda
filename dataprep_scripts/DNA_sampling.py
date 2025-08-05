import os
import sys
import glob
import argparse
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from select_genomes import get_gtdb_info


ncbi_datasets_exec = "/work/pi_yingzhang_uri_edu/ccres/tools/datasets"


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
	if f'{genome_id}' not in os.listdir(args.genomes_dir):
		output_dir = os.path.join(args.genomes_dir, f'{genome_id}')
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
    parser.add_argument('--genomes_dir', type=str, help='path to directory containing gtf annotations files')
    parser.add_argument('--species', type=str, help='species with GTDB taxonomy', choices=['Prochlorococcus_B marinus_B','Marinobacter psychrophilus','Alteromonas macleodii'])
    parser.add_argument('--gtdb_info', type=str, help='path to GTDB metadata file')
    parser.add_argument('--gtdb_genomes', type=str, help='path to directory containing GTDB genomes')
    args = parser.parse_args()

    # create output directory
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.isdir(args.genomes_dir):
        os.makedirs(args.genomes_dir)
    

    # get genomes from GTDB
    genomes = GetGenomes(args)
    # get fasta files and annotations
    for genome_id in genomes:
        print(genome_id)
        GetGenomeAndAnnot(args, genome_id)
    
    # run anvio

    # prepare training and validation datasets from one training genome, 

    

    # call dnabert script (provide the whole genome as input)

    # shuffle and split sequences between train and val (70/30)

    # create tfrecords

    # prepare testing dataset

    # get annotations

    # call dnabert script (provide the whole genome as input)