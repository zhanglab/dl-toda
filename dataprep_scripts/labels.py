import json
import os
import sys
from collections import defaultdict
import random
import math
import argparse


def create_dl_toda_tax(taxa2labels):
    with open('dl_toda_taxonomy.tsv', 'w') as outf:
        for k, v in taxa2labels.items():
            outf.write(f'{v}\t{k}\n')


def get_labels(rank_index, rank_name, taxa2labels):
    # create json dictionary
    labels2taxa = {v: k.split(';')[rank_index] for k, v in taxa2labels.items()}
    with open(f'{rank_name}_labels.json', 'w') as f:
        json.dump(labels2taxa, f)


def get_genomes(tax_file, rank_index):
    genomes = defaultdict(list)  # key = taxon, value = list of genomes
    with open(tax_file, 'r') as inf:
        for line in inf:
            genomes[";".join(line.rstrip().split('\t')[rank_index+1:7])].append(line.rstrip().split('\t')[0])
    # create dictionary mapping labels to taxa
    taxa = list(set(genomes.keys()))
    taxa2labels = dict(zip(taxa, list(range(len(taxa)))))  # key = taxon, value = label
    # split list of genomes into training and testing
    out_train = open('training_genomes.tsv', 'w')
    out_test = open('testing_genomes.tsv', 'w')
    for k, v in genomes.items():
        print(k, len(v))
        if len(v) > 1:
            random.shuffle(v)
            num_train_genomes = math.ceil(0.7*len(v))
            for g in v[:num_train_genomes]:
                out_train.write(f'{g}\t{taxa2labels[k]}\n')
            for g in v[num_train_genomes:]:
                out_test.write(f'{g}\t{taxa2labels[k]}\n')
        else:
            out_train.write(f'{v[0]}\t{taxa2labels[k]}\n')

    return taxa2labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tax_file', type=str, help='')
    parser.add_argument('--rank', type=str, help='taxonomic rank',
                        choices=['species', 'genus', 'family', 'order', 'class', 'phylum'])
    parser.add_argument('--dl_toda_tax', help='create dl_toda_taxonomy.tsv file', action='store_true')
    args = parser.parse_args()

    ranks = {'species': 0, 'genus': 1, 'family': 2, 'order': 3, 'class': 4, 'phylum': 5}

    # get genomes for training and testing
    taxa2labels = get_genomes(args.tax_file, ranks[args.rank])

    # create dictionary mapping labels to taxa at given rank
    get_labels(ranks[args.rank], args.rank, taxa2labels)

    if args.dl_toda_tax:
        create_dl_toda_tax(taxa2labels)


if __name__ == "__main__":
    main()
