from Bio import SeqIO
import pandas as pd
import glob
import os
import sys

def get_gtdb_info(gtdb_info):
    # load gtdb info file
    gtdb_df = pd.read_csv(gtdb_info, delimiter='\t', usecols=['accession', 'gtdb_genome_representative', 'gtdb_taxonomy', 'ncbi_genome_category', 'ncbi_assembly_level', 'ncbi_genome_representation'])
    genomes = [i[3:] for i in gtdb_df['accession'].tolist()]
    ncbi_assembly_level = gtdb_df['ncbi_assembly_level'].tolist()
    ncbi_genome_category = gtdb_df['ncbi_genome_category'].tolist()
    ncbi_genome_representation = gtdb_df['ncbi_genome_representation'].tolist()
    gtdb_rep_genome = [i[3:] for i in gtdb_df['gtdb_genome_representative'].tolist()]

    return genomes, ncbi_assembly_level, ncbi_genome_category, ncbi_genome_representation, gtdb_rep_genome

def clean_fasta(genome_id, fastafile, path_to_db, output_dir, out_f):
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
        out_f.write(f'{genome_id}\t{new_filepath}\n')


def get_genomes(path_to_db):
    # get fasta files in database
    fasta_files = glob.glob(os.path.join(path_to_db, '*.fna'))
    # map genomes accession id to path to fasta files
    genomes = {"_".join(i.split('/')[-1].split('_')[0:2]): i for i in fasta_files}

    return genomes


def main():
    gtdb_info = sys.argv[1]
    ncbi_refseq_db = sys.argv[2]
    gtdb_db = sys.argv[3]
    output_dir = sys.argv[4]

    # create directory to store cleaned fasta files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # parse gtdb info file (bac120_metadata_r95.tsv)
    genomes, ncbi_assembly_level, ncbi_genome_category, ncbi_genome_representation, gtdb_rep_genome = get_gtdb_info(gtdb_info)

    # get list of genomes available locally
    ncbi_genomes = get_genomes(ncbi_refseq_db)
    gtdb_genomes = get_genomes(gtdb_db)

    with open(os.path.join(output_dir, 'genomes.tsv'), 'w') as out_f:
        for i in range(len(genomes)):
            if ncbi_assembly_level[i] == "Complete Genome" and ncbi_genome_category[i] != "derived from metagenome" and ncbi_genome_category[i] != "derived from environmental_sample":
                print(ncbi_assembly_level[i], ncbi_genome_category[i], ncbi_genome_category[i])
                # clean fasta file
                if genomes[i] in ncbi_genomes:
                    clean_fasta(genomes[i], ncbi_genomes[genomes[i]], ncbi_refseq_db, output_dir, out_f)
                elif genomes[i] in gtdb_genomes:
                    clean_fasta(genomes[i], gtdb_genomes[genomes[i]], gtdb_db, output_dir, out_f)



if __name__ == "__main__":
    main()
