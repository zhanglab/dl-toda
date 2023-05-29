import os
import argparse
import shutil
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--taxon', type=str, help='taxonomic group of interest')
    parser.add_argument('--rank', type=str, help='taxonomic rank of taxon')
    parser.add_argument('--fq_dir', type=str, help='directory containing fastq files of simulated reads')
    parser.add_argument('--output_dir', type=str, help='output directory containing fq files of taxon of interest')
    args = parser.parse_args()

    ranks = {'species': 0, 'genus': 1, 'family': 2, 'order': 3, 'class': 4, 'phylum': 5}

    dl_toda_taxonomy = '/'.join(
        os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + f'/data/dl_toda_taxonomy.tsv'

    dl_toda_tax = {}
    with open(dl_toda_taxonomy, 'r') as f:
        # get all species under the given taxonomic group
        for line in f:
            r_taxon = line.rstrip().split('\t')[1].split(';')[ranks[args.rank]]
            if r_taxon == args.taxon:
                sp = line.rstrip().split('\t')[1].split(';')[0]
                dl_toda_tax[sp] = line.rstrip().split('\t')[0]

    # create json dictionary of
    labels2taxa = dict(zip(len(dl_toda_tax), list(dl_toda_tax.keys())))
    with open(os.path.join(args.fq_dir, f'{args.taxon}_species_labels.json'), 'w') as f:
        json.dump(labels2taxa, f)

    # create directory with fastq files of taxon of interest
    if not os.path.isdir(os.path.join(args.output_dir, f'{args.taxon}_data')):
        os.makedirs(os.path.join(args.output_dir, f'{args.taxon}_data'))

    # copy fq files to new directory with updated name
    for k_taxon, old_label in dl_toda_tax.items():
        for new_label, v_taxon in labels2taxa.items():
            if v_taxon == k_taxon:
                shutil.copy(os.path.join(args.fq_dir, f'{old_label}_training.fq'), os.path.join(args.output_dir, f'{args.taxon}_data', f'{new_label}_training.fq'))



