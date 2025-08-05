import os
import sys
import glob
import argparse
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from select_genomes import get_gtdb_info

def GetGenomes(args):
    genomes, ncbi_assembly_level, ncbi_genome_category, ncbi_genome_representation, gtdb_rep_genome, gtdb_taxonomy, ncbi_taxonomy = get_gtdb_info(args.gtdb_info)
    genus = args.species.split(' ')[0]
    print(genus)
    for i in range(len(genomes)):
        if gtdb_taxonomy[i].split(';')[-1].split('__')[1] == args.species or gtdb_taxonomy[i].split(';')[-2].split('__')[1] == genus:
            if ncbi_assembly_level[i] == "Complete Genome" and ncbi_genome_category[i] != "derived from metagenome" and ncbi_genome_category[i] != "derived from environmental_sample":
                print(genomes[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    parser.add_argument('--species', type=str, help='species with GTDB taxonomy', choices=['Prochlorococcus_B marinus_B','Marinobacter psychrophilus','Alteromonas macleodii'])
    parser.add_argument('--gtdb_info', type=str, help='path to GTDB metadata file')
    parser.add_argument('--gtdb_genomes', type=str, help='path to directory containing GTDB genomes')
    args = parser.parse_args()

    # create output directory
    if not os.path.isdir(args.output_dir):
		os.makedirs(args.output_dir)
    if not os.path.isdir(os.path.join(args.output_dir, 'fasta')):
		os.makedirs(os.path.join(args.output_dir, 'fasta'))

    # get genomes from GTDB
    genomes = GetGenomes(args)
    
    # run anvio

    # prepare training and validation datasets from training genome

    # get annotations

    # call dnabert script (provide the whole genome as input)

    # shuffle and split sequences between train and val (70/30)

    # create tfrecords

    # prepare testing dataset

    # get annotations

    # call dnabert script (provide the whole genome as input)