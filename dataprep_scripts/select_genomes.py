from Bio import SeqIO
import pandas as pd
import glob
import os
import sys
import argparse

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

def clean_fasta(genome_id, fastafile, path_to_db, output_dir, outf):
    updated_seq = []
    updated_description = []
    for record in SeqIO.parse(fastafile, "fasta"):
        # remove phages and plasmids
        if 'plasmid' not in record.description and 'Plasmid' not in record.description and 'phage' not in record.description:
            updated_seq.append(str(record.seq))
            updated_description.append(record.description)
    # only keep genomes with size equal or above 500000 bp
    if len("".join(updated_seq)) >= 500000:
        # if more than one chromosome, combine chromosomes into one sequence
        new_description = f'{updated_description[0]}, combined' if len(updated_description) > 1 else updated_description[0]
        new_filepath = os.path.join(output_dir, f'updated_{fastafile.split("/")[-1]}')
        print(new_filepath)
        with open(new_filepath, 'w') as out_fasta:
            out_fasta.write(f'>{new_description}\n{"".join(updated_seq)}\n')
        outf.write(f'{genome_id}\t{new_filepath}\n')


def get_genomes(path_to_db):
    # get fasta files in database
    fasta_files = glob.glob(os.path.join(path_to_db, '*.fna'))
    # map genomes accession id to path to fasta files
    genomes = {"_".join(i.split('/')[-1].split('_')[0:2]): i for i in fasta_files}

    return genomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gtdb_info', type=str, help='path to bac120_metadata_r220.tsv file')
    parser.add_argument('--ncbi_refseq_db', type=str, help='path to ncbi refseq database')
    parser.add_argument('--gtdb_db', type=str, help='path to gtdb database')
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    parser.add_argument('--used_genomes', type=str, help='file containing list of genomes already used for training or testing')
    parser.add_argument('--labels', nargs='+', help='list of labels in dltoda')
    args = parser.parse_args()

    # create directory to store cleaned fasta files
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # parse gtdb info file (bac120_metadata_r95.tsv)
    genomes_id, ncbi_assembly_level, ncbi_genome_category, ncbi_genome_representation, gtdb_rep_genome, gtdb_taxonomy, ncbi_taxonomy = get_gtdb_info(args.gtdb_info)

    # get list of genomes available locally
    ncbi_genomes = get_genomes(args.ncbi_refseq_db)
    gtdb_genomes = get_genomes(args.gtdb_db)

    if args.used_genomes is None:
        used_genomes = []
    else:
        with open(args.used_genomes, 'r') as f:
            used_genomes = [line.rstrip() for line in f.readlines()]

    # retrieve gtdb taxonomy of species in dltoda
    path_dl_toda_tax = '/'.join(
                os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/data/dl_toda_taxonomy.tsv'
    dl_toda_tax = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[1] for line in content}
    
    with open(os.path.join(output_dir, 'genomes.tsv'), 'w') as outf:
        for i in range(len(genomes_id)):
            if genomes_id[i] not in used_genomes:
                if ncbi_assembly_level[i] == "Complete Genome" and ncbi_genome_category[i] != "derived from metagenome" and ncbi_genome_category[i] != "derived from environmental_sample":
                    outf.write(f'{genomes_id[i]}\t{gtdb_taxonomy[i]}\t{ncbi_assembly_level[i]}\t{ncbi_genome_category[i]}\t{ncbi_genome_representation[i]}\t{gtdb_rep_genome[i]}\t')
                    # clean fasta file
                    if genomes_id[i] in ncbi_genomes:
                        outf.write(f'NCBI\n')
                        # clean_fasta(genomes[i], ncbi_genomes[genomes[i]], ncbi_refseq_db, output_dir, outf)
                    elif genomes_id[i] in gtdb_genomes:
                        # clean_fasta(genomes[i], gtdb_genomes[genomes[i]], gtdb_db, output_dir, outf)
                        outf.write(f'GTDB\n')
                    else:
                        outf.write(f'NOT IN\n')




if __name__ == "__main__":
    main()
