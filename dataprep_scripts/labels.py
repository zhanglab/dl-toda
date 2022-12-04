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
    labels2taxa = {v: k.split(';')[rank_index] for k, v in taxa2labels.items()}
    with open('labels.json', 'w') as f:
        json.dump(labels2taxa, f)

def get_genomes(tax_file, rank_index):
    genomes = defaultdict(list) # key = taxon, value = list of genomes
    with open(tax_file, 'r') as inf:
        for line in inf:
            genomes[";".join(line.rstrip().split('\t')[rank_index+1:7])].append(line.rstrip().split('\t')[0])
    # create dictionary mapping labels to taxa
    taxa = list(set(genomes.keys()))
    taxa2labels = dict(zip(taxa,list(range(len(taxa))))) # key = taxon, value = label
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
    tax_file = sys.argv[1] # tsv file with genome and their assigned gtdb or ncbi taxonomy
    rank = sys.argv[2] # rank

    ranks = {'species': 0, 'genus': 1, 'family': 2, 'order': 3, 'class': 4, 'phylum': 5}

    # get genomes for training and testing
    taxa2labels = get_genomes(tax_file, ranks[rank])

    # create dictionary mapping labels to taxa at given rank
    get_labels(tax_file, ranks[rank], taxa2labels)












if __name__ == "__main__":
    main()
