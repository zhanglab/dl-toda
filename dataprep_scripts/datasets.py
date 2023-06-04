import sys
import os
import random
import math
import json
from collections import defaultdict
import shutil
import glob
import multiprocessing as mp
import argparse


def load_fq_file(args, fq_file):
    with open(fq_file, 'r') as f:
        content = f.readlines()
        reads = [''.join(content[j:j + args.num_lines]) for j in range(0, len(content), args.num_lines)]
        return reads


def create_sets(args, reads, set_type, output_dir):
    # get total number of reads
    list_num_reads = []
    for process, process_reads in reads.items():
        list_num_reads.append(process_reads)
    num_reads = sum(list_num_reads)
    print(num_reads)
    if args.balanced:
        num_sets = 5
        # get minimum number of reads per species in list
        min_num_reads = min(list_num_reads)
        print(f'min number of reads: {min_num_reads}')
        # compute number of reads per species and per set
        num_reads_per_set = min_num_reads // num_sets
        print(f'# reads per species and per set: {num_reads_per_set}\ntotal # reads: {min_num_reads * len(args.taxa)}')
    else:
        # compute number of sets given that we want 20000000 reads per set
        num_sets = math.ceil(num_reads / 20000000) if num_reads > 20000000 else 1
    print(num_sets)
    label_out = open(os.path.join(output_dir, f'{set_type}-label-read-count'), 'w')
    for label in args.taxa:
        label_fq_files = sorted(glob.glob(os.path.join(output_dir, set_type, 'reads-*', f'{set_type}-{label}.fq')))
        list_reads = []
        for fastq in label_fq_files:
            list_reads = load_fq_file(args, fastq)
        random.shuffle(list_reads)
        print(f'label: {label}\t# reads: {len(list_reads)}')
        if args.balanced:
            # updated list of reads
            list_reads = list_reads[:min_num_reads]
        else:
            # compute number of reads for given species per set
            num_reads_per_set = math.ceil(len(list_reads)/num_sets)

        label_out.write(f'{label}\t{len(list_reads)}\t{num_reads_per_set}\n')
        for count, i in enumerate(range(0, len(list_reads), num_reads_per_set)):
            with open(os.path.join(output_dir, f'{set_type}-subset-{count}.fq'), 'a') as outfile:
                outfile.write(''.join(list_reads[i:i+num_reads_per_set]))


# def split_reads(grouped_genomes, input_dir, output_dir, genomes2labels, taxa2labels, process_id, train_reads, val_reads):
def split_reads(args, grouped_files, output_dir, process_id, train_reads, val_reads):
    # create directories to store output fq files
    process_train_reads = 0
    process_val_reads = 0
    if not os.path.exists(os.path.join(output_dir, 'train', f'reads-{process_id}')):
        os.makedirs(os.path.join(output_dir, 'train', f'reads-{process_id}'))
    if not os.path.exists(os.path.join(output_dir, 'val', f'reads-{process_id}')):
        os.makedirs(os.path.join(output_dir, 'val', f'reads-{process_id}'))

    for fq_file in grouped_files:
        label = fq_file.split('/')[-1].split('_')[0]
        reads = load_fq_file(args, fq_file)
        random.shuffle(reads)
        num_train_reads = math.ceil(0.7*(len(reads)))
        with open(os.path.join(output_dir, 'train', f'reads-{process_id}', f'train-{label}.fq'), 'w') as out_f:
            out_f.write(''.join(reads[:num_train_reads]))
        with open(os.path.join(output_dir, 'val', f'reads-{process_id}', f'val-{label}.fq'), 'w') as out_f:
            out_f.write(''.join(reads[num_train_reads:]))
        process_train_reads += num_train_reads
        process_val_reads += len(reads) - num_train_reads

    train_reads[process_id] = process_train_reads
    val_reads[process_id] = process_val_reads


# def create_train_val_sets(input_dir, output_dir, genomes2labels, taxa2labels):
def create_train_val_sets(args):
    # get list of fastq files for training
    fq_files = glob.glob(os.path.join(args.input_dir, "*_training.fq"))
    args.taxa = [i.split('/')[-1].split('_')[0] for i in fq_files]
    print(len(fq_files), len(args.taxa))
    chunk_size = math.ceil(len(fq_files)/mp.cpu_count())
    grouped_files = [fq_files[i:i+chunk_size] for i in range(0, len(fq_files), chunk_size)]
    with mp.Manager() as manager:
        train_reads = manager.dict()
        val_reads = manager.dict()
        processes = [mp.Process(target=split_reads, args=(args, grouped_files[i], args.output_dir, i, train_reads, val_reads)) for i in range(len(grouped_files))]
        # processes = [mp.Process(target=split_reads, args=(grouped_genomes[i], input_dir, output_dir, genomes2labels, taxa2labels, i, train_reads, val_reads)) for i in range(len(grouped_genomes))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        create_sets(args, train_reads, 'train', args.output_dir)
        create_sets(args, val_reads, 'val', args.output_dir)

    # remove temporary directories
    shutil.rmtree(os.path.join(args.output_dir, 'train'))
    shutil.rmtree(os.path.join(args.output_dir, 'val'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help="Path to directory with fastq files of simulated reads")
    parser.add_argument('--output_dir', help="Path to the output directory")
    parser.add_argument('--pair', action='store_true', default=False, help="process reads as pairs")
    parser.add_argument('--balanced', action='store_true', default=False, help="have each species evenly represented per subsets")
    args = parser.parse_args()

    args.num_lines = 8 if args.pair else 4

    if not os.path.exists(os.path.join(args.output_dir, 'train')):
        os.makedirs(os.path.join(args.output_dir, 'train'))
    if not os.path.exists(os.path.join(args.output_dir, 'val')):
        os.makedirs(os.path.join(args.output_dir, 'val'))
    create_train_val_sets(args)


if __name__ == "__main__":
    main()
