from Bio import SeqIO
import pandas as pd
import glob
import os
import sys
import argparse
import gzip
import shutil

def get_gtdb_info(gtdb_info):
    # load gtdb info file
    gtdb_df = pd.read_csv(gtdb_info, delimiter='\t', usecols=['accession', 'gtdb_genome_representative', 'gtdb_taxonomy', 'ncbi_taxonomy', 'ncbi_genome_category', 'ncbi_assembly_level', 'ncbi_genome_representation'])
    genomes = [i[3:] for i in gtdb_df['accession'].tolist()]
    ncbi_assembly_level = gtdb_df['ncbi_assembly_level'].tolist()
    ncbi_genome_category = gtdb_df['ncbi_genome_category'].tolist()
    ncbi_genome_representation = gtdb_df['ncbi_genome_representation'].tolist()
    gtdb_rep_genome = [i[3:] for i in gtdb_df['gtdb_genome_representative'].tolist()]
    gtdb_taxonomy = gtdb_df['gtdb_taxonomy'].tolist()
    ncbi_taxonomy = gtdb_df['ncbi_taxonomy'].tolist()

    return genomes, ncbi_assembly_level, ncbi_genome_category, ncbi_genome_representation, gtdb_rep_genome, gtdb_taxonomy, ncbi_taxonomy

def clean_fasta(args, fasta_file):
    updated_seq = []
    updated_description = []

    if fasta_file[-2:] == 'gz':
        with gzip.open(fasta_file, 'rt') as handle:
            for record in SeqIO.parse(handle, "fasta"):
                # remove phages and plasmids
                if 'plasmid' not in record.description and 'Plasmid' not in record.description and 'phage' not in record.description:
                    updated_seq.append(str(record.seq))
                    updated_description.append(record.description)
    else:
        for record in SeqIO.parse(fasta_file, "fasta"):
            # remove phages and plasmids
            if 'plasmid' not in record.description and 'Plasmid' not in record.description and 'phage' not in record.description:
                updated_seq.append(str(record.seq))
                updated_description.append(record.description)

    # only keep genomes with size equal or above 500000 bp
    if len("".join(updated_seq)) >= 500000:
        # if more than one chromosome, combine chromosomes into one sequence
        new_description = f'{updated_description[0]}, combined' if len(updated_description) > 1 else updated_description[0]
        new_filepath = os.path.join(args.output_dir, 'cleaned_genomes', f'updated_{fasta_file.split("/")[-1]}')
        print(new_filepath)
        with open(new_filepath, 'w') as out_fasta:
            out_fasta.write(f'>{new_description}\n{"".join(updated_seq)}\n')


def get_fasta(args, genomes_id):
    genomes_to_fa = {}
    with open(args.ncbi_refseq_db, 'r') as f:
        for line in f:
            genome = '_'.join(line.rstrip().split('/')[-2].split('_')[0:2])
            genomes_to_fa[genome] = line.rstrip()

    for genome in genomes_id:
        genome = genome.replace('_', '').split('.')[0]
        path = os.path.join(args.output_dir, '/'.join([genome[i:i+3] for i in range(0,10,3)]))
        if genome not in genomes_to_fa:
            genomes_to_fa[genome] = path

    return genomes_to_fa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gtdb_info', type=str, help='path to bac120_metadata_r220.tsv file')
    parser.add_argument('--ncbi_refseq_db', type=str, help='path to file containing list of fasta files in ncbi refseq db') # /datasets/bio/ncbi-refseq/bacterial_genomes
    parser.add_argument('--gtdb_db', type=str, help='path to local gtdb database') # /datasets/bio/gtdb/release220/genomic_files_reps/gtdb_genomes_reps_r220/database/list-fna-files
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    parser.add_argument('--used_genomes', type=str, help='file containing list of genomes already used for training or testing')
    parser.add_argument('--labels', type=str, help='file with list of labels in dltoda')
    args = parser.parse_args()

    # create directory to store cleaned fasta files
    if not os.path.exists(os.path.join(args.output_dir, 'original_genomes')):
        os.makedirs(os.path.join(args.output_dir, 'original_genomes'))
    if not os.path.exists(os.path.join(args.output_dir, 'cleaned_genomes')):
        os.makedirs(os.path.join(args.output_dir, 'cleaned_genomes'))

    # parse gtdb info file (bac120_metadata_r95.tsv)
    genomes_id, ncbi_assembly_level, ncbi_genome_category, ncbi_genome_representation, gtdb_rep_genome, gtdb_taxonomy, ncbi_taxonomy = get_gtdb_info(args.gtdb_info)

    # get list of genomes available locally
    genomes_to_fa = get_fasta(args, genomes_id)

    if 'GCF_000195975.1' in genomes_to_fa:
        print(f'799\tGCF_000195975.1\t{genomes_to_fa["GCF_000195975.1"]}')

    if 'GCF_000013785.1' in genomes_to_fa:
        print(f'282\tGCF_000013785.1\t{genomes_to_fa["GCF_000013785.1"]}')

    if args.used_genomes is None:
        used_genomes = []
    else:
        with open(args.used_genomes, 'r') as f:
            used_genomes = [line.rstrip() for line in f.readlines()]

    # retrieve gtdb taxonomy of species in dltoda
    path_dl_toda_tax = '/'.join(
                os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/data/dl_toda_taxonomy.tsv'
    with open(path_dl_toda_tax, 'r') as f:
        dl_toda_tax = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[1] for line in f.readlines()}

    outf = open(os.path.join(args.output_dir, 'genomes.tsv'), 'w')
    with open(args.labels, 'r') as f:
        for label in f:
            label = label.rstrip()
            species = dl_toda_tax[label].split(';')[0]
            genus = dl_toda_tax[label].split(';')[1]
            print(label, species, genus)
            for i in range(len(genomes_id)):
                if genomes_id[i] in ['GCF_000195975.1', 'GCF_000013785.1']:
                    print(gtdb_taxonomy[i], ncbi_assembly_level[i], ncbi_genome_category[i])
                if gtdb_taxonomy[i].split(';')[-1].split('__')[1] == species or gtdb_taxonomy[i].split(';')[-2].split('__')[1] == genus:
                    if genomes_id[i] not in used_genomes:
                        print(genomes_id[i])
                        if ncbi_assembly_level[i] == "Complete Genome" and ncbi_genome_category[i] != "derived from metagenome" and ncbi_genome_category[i] != "derived from environmental_sample":
                            line = f'{label}\t'
                            if gtdb_taxonomy[i].split(';')[-1].split('__')[1] == species:
                                line += '1\t'
                            else:
                                if gtdb_taxonomy[i].split(';')[-2].split('__')[1] == genus:
                                    line += '0\t'
                            line += f'{genomes_id[i]}\t{gtdb_taxonomy[i]}\t{ncbi_assembly_level[i]}\t{ncbi_genome_category[i]}\t{ncbi_genome_representation[i]}\t{gtdb_rep_genome[i]}\t'
                            if genomes_id[i] in genomes_to_fa:
                                line += 'IN\n'
                                # copy fasta file to output directory
                                source_path = genomes_to_fa[genomes_id[i]]
                                fasta_filename = source_path.split('/')[-1]
                                dest_path = os.path.join(args.output_dir, 'original_genomes', fasta_filename)    
                                shutil.copy(source_path, dest_path)
                                clean_fasta(args, dest_path)
                            else:
                                line += 'NOT IN\n'
                                
                            outf.write(line)


if __name__ == "__main__":
    main()
