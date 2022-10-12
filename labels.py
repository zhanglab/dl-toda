import json
import os
import sys
from collections import defaultdict
import random
import math

def get_labels(tax_file, rank_index, taxa2labels):
    with open('dl_toda_taxonomy.tsv', 'w') as outf:
        for k, v in taxa2labels.items():
            outf.write(f'{v}\t{k}\n')

    # create json dictionary
    labels2taxa = {v: k.split(';')[rank_index-1] for k, v in taxa2labels.items()}
    with open('labels.json', 'w') as f:
        json.dump(labels2taxa, f)

def get_genomes(tax_file):
    genomes = defaultdict(list) # key = taxon, value = list of genomes
    with open(tax_file, 'r') as inf:
        for line in inf:
            genomes[";".join(line.rstrip().split('\t')[1:])].append(line.rstrip().split('\t')[0])
    # create dictionary mapping labels to taxa
    taxa = list(set(genomes.keys()))
    taxa2labels = dict(zip(taxa,list(range(len(taxa))))) # key = taxon, value = label
    # split list of genomes into training and testing
    out_train = open('training_genomes.tsv', 'w')
    out_test = open('testing_genomes.tsv', 'w')
    for k, v in genomes.items():
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
    tax_file = sys.argv[1] # tsv file with genome and their assigned gtdb taxonomy
    rank = sys.argv[2] # rank

    ranks = {'species': 1, 'genus': 2, 'family': 3, 'order': 4, 'class': 5, 'phylum': 6}

    # get genomes for training and testing
    taxa2labels = get_genomes(tax_file)

    # create dictionary mapping labels to taxa at given rank
    get_labels(tax_file, ranks[rank], taxa2labels)












if __name__ == "__main__":
    main()
