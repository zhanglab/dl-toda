import argparse
import sys
import os
import re
import json
import math
import multiprocessing as mp
import pandas as pd
import gzip
from collections import defaultdict
from dl_toda_tax_utils import get_rank_taxa

def load_reads(args):
    # with gzip.open(args.fastq, 'rt') as handle:
    with open(args.fastq, 'r') as handle:
        content = handle.readlines()
        records = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
        reads = {i.split("\n")[0]: i for i in records}

    return reads

def bin_reads(data, args, results, process_id):
    process_results = defaultdict(list) # key = taxon, value = reads
    for line in data:
        print(line)
        score = float(line.split('\t')[3])
        if int(line.split('\t')[2]) in args.labels_mapping_dict[args.rank]:
            taxon = args.labels_mapping_dict[args.rank][int(line.split('\t')[2])]
            if score > args.cutoff:
                process_results[taxon].append(line.split('\t')[0])
    results[process_id] = process_results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dl_toda_output', type=str, help='output file with predicted species obtained from running DL-TODA')
    parser.add_argument('--fastq', type=str, help='path to sample gzipped fastq file')
    parser.add_argument('--processes', type=int, help='number of processes')
    parser.add_argument('--tool', type=str, help='taxonomic classification tool used', choices=['dl-toda', 'kraken'])
    parser.add_argument('--output_dir', type=str, help='path to output directory', default=os.getcwd())
    parser.add_argument('--rank', type=str, help='taxonomic rank at which the binning should be done')
    parser.add_argument('--dl_toda_tax', type=str, help='path to DL-TODA taxonomy')
    parser.add_argument('--cutoff', type=float, help='cutoff or real value between 0 and 1 at which reads should be binned', default=0.0)
    parser.add_argument('--taxa', nargs='+', default=[], help='list of taxa to bin')
    parser.add_argument('--all', help='bin all taxa', action='store_true')
    parser.add_argument('--tax_db', help='type of taxonomy database used in DL-TODA', choices=['ncbi', 'gtdb'])
    parser.add_argument('--hist', help='provide histogram of confidence scores', action='store_true')
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.dl_toda_output.split('/')[-1].split('-')[0], f'cutoff-{args.cutoff}') if len(args.dl_toda_output.split('/')[-1].split('-')) == 2 else os.path.join(args.output_dir, '-'.join(args.dl_toda_output.split('/')[-1].split('-')[0:3]), f'cutoff-{args.cutoff}')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.ranks = {'phylum': 5, 'class': 4, 'order': 3, 'family': 2, 'genus': 1, 'species': 0}

    # get dl_toda taxonomy
    if args.tax_db == 'ncbi':
        index = 2
    elif args.tax_db =='gtdb':
        index = 3
    with open(os.path.join(args.dl_toda_tax, 'dl_toda_taxonomy.tsv'), 'r') as in_f:
        content = in_f.readlines()
        args.dl_toda_taxonomy = {}
        for i in range(len(content)):
            line = content[i].rstrip().split('\t')
            args.dl_toda_taxonomy[int(line[1])] = line[index]

    get_rank_taxa(args, args.dl_toda_taxonomy)

    # get reads sequences
    reads = load_reads(args)

    print(f'file: {args.dl_toda_output}')
    print(f'#reads: {len(reads)}')

    with open(args.dl_toda_output, 'r') as f:
        content = f.readlines()
    chunk_size = math.ceil(len(content)/args.processes)
    data_split = [content[i:i+chunk_size] for i in range(0,len(content),chunk_size)]
    print(f'{len(data_split)}\t{args.processes}\t{chunk_size}\t{len(data_split[0])}')

    with mp.Manager() as manager:
        results = manager.dict()
        # Create new processes
        processes = [mp.Process(target=bin_reads, args=(data_split[i], args, results, i)) for i in range(len(data_split))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        summary = open(os.path.join(args.output_dir, 'summary.tsv'), 'w')
        for t in args.labels_mapping_dict[args.rank].values():
            print(t)
            filename, _ = re.subn('[/,(,)]', ' ', t)
            print(filename)
            with open(os.path.join(args.output_dir, "-".join(filename.split(" ")) + '-bin.fq'), 'w') as out_f:
                n_reads = 0
                for process_id, process_results in results.items():
                    if t in process_results:
                        t_reads = [reads[i] for i in process_results[t]]
                        out_f.write(f'{"".join(t_reads)}')
                        n_reads += len(t_reads)
                summary.write(f'{t}\t{n_reads}\n')
