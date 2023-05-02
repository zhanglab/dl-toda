import json
import os
import sys
from collections import defaultdict
import argparse
from labels import get_labels


def map_taxa2labels(args, ranks):
    # define index of taxonomic database
    index = 1 if args.tax_db == 'gtdb' else 2

    path_dl_toda_tax = '/'.join(
        os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/data/dl_toda_taxonomy.tsv'

    taxa = set()
    with open(path_dl_toda_tax, 'r') as in_f:
        content = in_f.readlines()
        # get taxa
        gtdb_taxonomy = {line.rstrip().split('\t')[index].split(';')[ranks[args.rank]]: ';'.join(line.rstrip().split("\t")[1].split(';')[ranks[args.rank]:6]) for
                         line in content}
        ncbi_taxonomy = {line.rstrip().split('\t')[index].split(';')[ranks[args.rank]]: ';'.join(line.rstrip().split("\t")[2].split(';')[ranks[args.rank]:6]) for
                         line in content}
        print(len(gtdb_taxonomy), len(ncbi_taxonomy))
        taxa2labels = dict(zip(gtdb_taxonomy.keys(), list(range(len(gtdb_taxonomy)))))  # key = taxon, value = label
        print(taxa2labels)
        # create dictionary mapping labels to taxa
        get_labels(ranks[args.rank], args.rank, taxa2labels)
        # create file mapping species labels to given rank labels
        with open(f'species_to_{args.rank}', 'w') as out_f:
            for line in content:
                line = line.rstrip().split('\t')
                taxon = line[index].split(';')[ranks[args.rank]]
                out_f.write(f'{line[0]}\t{taxa2labels[taxon]}\n')
        # create file mapping rank labels to gtdb and ncbi taxonomy
        with open(f'dl_toda_taxonomy_{args.rank}.tsv') as out_f:
            for k, v in gtdb_taxonomy.items():
                out_f.write(f'{taxa2labels[k]}\t{v}\t{ncbi_taxonomy[k]}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=str, help='taxonomic rank',
                        choices=['species', 'genus', 'family', 'order', 'class', 'phylum'])
    parser.add_argument('--tax_db', type=str, help='taxonomic database',
                        choices=['gtdb', 'ncbi'])
    args = parser.parse_args()

    ranks = {'species': 0, 'genus': 1, 'family': 2, 'order': 3, 'class': 4, 'phylum': 5}

    map_taxa2labels(args, ranks)


if __name__ == "__main__":
    main()