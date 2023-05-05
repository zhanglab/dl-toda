import os
import argparse
import json
import random
from collections import defaultdict
from Bio import SeqIO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='path to output directory')
    parser.add_argument('--output_dir', type=str, help='path to output directory', default=os.getcwd())
    parser.add_argument('--dataset', type=str, help='type of dataset', choices=['train','val'])
    parser.add_argument('--n_reads', type=int, help='number of reads per taxon')
    parser.add_argument('--class_mapping', type=str, help='path to json dictionary mapping labels to taxa')
    args = parser.parse_args()

    # load class_mapping file mapping label IDs to species
    f = open(args.class_mapping)
    class_mapping = json.load(f)

    fq_files = os.listdir(os.getcwd())
    random.shuffle(fq_files)
    taxa_count = defaultdict(int)
    n_reads_per_tax = 20000000 // len(class_mapping)
    count_files = 0
    count_reads = 0
    for i in range(len(fq_files)):
        with open(fq_files[i], 'r') as handle:
            rec = ''
            n_line = 0
            for line in handle:
                rec += line
                n_line += 1
                if n_line == 4:
                    label = rec.split('\n')[0].rstrip().split('|')[1]
                    if taxa_count[label] < n_reads_per_tax:
                        with open(os.path.join(args.output_dir, f'{args.dataset}-subset-{count_files}.fq'), 'a') as out_f:
                            out_f.write(rec)
                        taxa_count[label] += 1
                        count_reads += 1
                        if count_reads == n_reads_per_tax*len(class_mapping):
                            count_files += 1
                            taxa_count = defaultdict(int)
                    # initialize variables again
                    n_line = 0
                    rec = ''


if __name__ == "__main__":
    main()