import sys
import pandas as pd


def main():
    gtdb_info =  sys.argv[1] # path to bac120_metadata_r95.tsv file provided with GTDB database
    file_w_genomes = sys.argv[2] # file with list of genome accession ids (1 id per line)
    # load and parse gtdb information file
    gtdb_df = pd.read_csv(gtdb_info, delimiter='\t', low_memory=False)
    accession_id = [i[3:] for i in list(gtdb_df.accession)]
    gtdb_tax = dict(zip(accession_id, list(gtdb_df.gtdb_taxonomy)))
    # retrieve gtdb taxonomy for every genome of interest
    with open('genomes-gtdb-taxonomy.tsv', 'w') as out_f:
        with open(file_w_genomes, 'r') as in_f:
            for line in in_f:
                genome = line.rstrip().split('\t')[0]
                taxa = [i.split('__')[1] for i in gtdb_tax[genome].split(';')[1:][::-1]]
                out_f.write(f'{genome}')
                for i in range(len(taxa)):
                    out_f.write(f'\t{taxa[i]}')
                out_f.write('\n')


if __name__ == "__main__":
    main()
