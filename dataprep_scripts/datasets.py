import os
import sys
import random
import math
import shutil
import glob
import multiprocessing as mp
import argparse
from utils import load_fq_file


def create_sets(args, reads, set_type, output_dir):
    # get total number of reads
    list_num_reads = []
    for process, process_reads in reads.items():
        list_num_reads += process_reads
    num_reads = sum(list_num_reads)
    print(f'# reads in {set_type} set: {num_reads}\t# taxa: {len(list_num_reads)}')
    if args.balanced:
        num_sets = 5
        if not args.num_reads:
            # get minimum number of reads per species in list
            args.num_reads = min(list_num_reads)
        print(f'# reads per species: {args.num_reads}')
        # compute number of reads per species and per set
        num_reads_per_set = min_num_reads // num_sets
        print(f'# reads per species and per set: {num_reads_per_set}\ntotal # reads: {min_num_reads * len(args.taxa)}')
    else:
        # compute number of sets given that we want 20000000 reads per set
        num_sets = math.ceil(num_reads / args.num_reads_per_set) if num_reads > args.num_reads_per_set else 1
    print(num_sets)
    label_out = open(os.path.join(output_dir, f'{set_type}-label-read-count'), 'w')
    for label in args.taxa:
        label_fq_files = sorted(glob.glob(os.path.join(output_dir, set_type, 'reads-*', f'{set_type}-{label}.fq')))
        list_reads = []
        for fastq in label_fq_files:
            list_reads = load_fq_file(fastq, args.num_lines)
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


def split_reads(args, grouped_files, output_dir, process_id, train_reads, val_reads):
    # create directories to store output fq files
    process_train_reads = []
    process_val_reads = []
    if not os.path.exists(os.path.join(output_dir, 'train', f'reads-{process_id}')):
        os.makedirs(os.path.join(output_dir, 'train', f'reads-{process_id}'))
    if not os.path.exists(os.path.join(output_dir, 'val', f'reads-{process_id}')):
        os.makedirs(os.path.join(output_dir, 'val', f'reads-{process_id}'))

    for fq_file in grouped_files:
        label = fq_file.split('/')[-1].split('_')[0]
        reads = load_fq_file(fq_file, args.num_lines)
        random.shuffle(reads)
        num_train_reads = math.ceil(0.7*(len(reads)))
        with open(os.path.join(output_dir, 'train', f'reads-{process_id}', f'train-{label}.fq'), 'w') as out_f:
            out_f.write(''.join(reads[:num_train_reads]))
        with open(os.path.join(output_dir, 'val', f'reads-{process_id}', f'val-{label}.fq'), 'w') as out_f:
            out_f.write(''.join(reads[num_train_reads:]))
        process_train_reads.append(num_train_reads)
        process_val_reads.append(len(reads) - num_train_reads)

    train_reads[process_id] = process_train_reads
    val_reads[process_id] = process_val_reads


def create_train_val_sets(args):
    if os.path.isdir(args.input):
        # get list of fastq files for training if input is directory
        fq_files = glob.glob(os.path.join(args.input_dir, "*_training.fq"))
    else:
        fq_files = [args.input]
        # update output directory
        args.output_dir = os.path.join(args.output_dir, args.input[:-3])

    if not os.path.exists(os.path.join(args.output_dir, 'train')):
        os.makedirs(os.path.join(args.output_dir, 'train'))
    if not os.path.exists(os.path.join(args.output_dir, 'val')):
        os.makedirs(os.path.join(args.output_dir, 'val'))

    args.taxa = [i.split('/')[-1].split('_')[0] for i in fq_files]
    print(len(fq_files), len(args.taxa))
    chunk_size = math.ceil(len(fq_files)/mp.cpu_count())
    grouped_files = [fq_files[i:i+chunk_size] for i in range(0, len(fq_files), chunk_size)]
    with mp.Manager() as manager:
        train_reads = manager.dict()
        val_reads = manager.dict()
        processes = [mp.Process(target=split_reads, args=(args, grouped_files[i], args.output_dir, i, train_reads, val_reads)) for i in range(len(grouped_files))]
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
    parser.add_argument('--input', help="Path to directory with fastq files or fastq file of simulated reads")
    parser.add_argument('--output_dir', help="Path to the output directory")
    parser.add_argument('--num_reads', type=int, help="number of reads per species")
    parser.add_argument('--num_reads_per_set', type=int, help="number of reads per set", default=20000000)
    parser.add_argument('--pair', action='store_true', default=False, help="process reads as pairs")
    parser.add_argument('--balanced', action='store_true', default=False, help="have each species evenly represented per subsets")
    args = parser.parse_args()

    args.num_lines = 8 if args.pair else 4        

    create_train_val_sets(args)


if __name__ == "__main__":
    main()
