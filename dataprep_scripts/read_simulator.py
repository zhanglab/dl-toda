import os
import sys
import glob
import argparse
import math
import random
import multiprocessing as mp
from Bio import SeqIO
import pandas as pd


def compute_size(fasta_file):
    new_seq = ''
    for record in SeqIO.parse(fasta_file, "fasta"):
        new_seq += record.seq
    return new_seq


def get_genomes_size(args, genomes):
    fasta_seq = {}
    for d in args.fasta_dir:
        list_fa = glob.glob(f'{d}/*.fna')
        for fa in list_fa:
            if '_'.join(fa.split('/')[-1].split('_')[2:4]) in genomes:
                fasta_seq['_'.join(fa.split('/')[-1].split('_')[2:4])] = compute_size(fa)
    # compute average size of selected genomes
    avg_size = sum([len(fasta_seq[g]) for g in genomes])/len(genomes)
    # get the largest size of all genomes
    max_size = max([len(fasta_seq[g]) for g in genomes])
    return fasta_seq, avg_size, max_size


def get_genomes(args):
    file_w_genomes = '/'.join(
        os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + f'/data/{args.dataset}_genomes.tsv'
    genomes = []
    with open(file_w_genomes, 'r') as f:
        for line in f:
            if line.rstrip().split('\t')[1] == str(args.label):
                genomes.append(line.rstrip().split('\t')[0])
    return genomes


def get_reverse(args, sequence, comp=False):
    # return reverse sequence (positive/upper strand in the 3' to 5' end direction)
    rev_seq = sequence[::-1]
    if comp:
        # return reverse complement sequence (negative/lower strand in the 5' to 3' end direction)
        rev_seq = ''.join([args.base_pairs[b] if b in args.base_pairs else b for b in rev_seq])
    return rev_seq


def replace_base(sequence, site, base):
    new_sequence = list(sequence)
    new_sequence[site] = base
    return "".join(new_sequence)


def simulate_reads(args, genomes):
    # compute average and max genome size
    fasta_seq, avg_size, max_size = get_genomes_size(args, genomes)
    # compute number of reads to simulate
    coverage = args.coverage if args.dataset == 'training' else 3
    avg_read_length = sum([100, 150, 250])/3
    # get number of pairs of reads to simulate
    n_reads = math.ceil(coverage * avg_size / (avg_read_length*2))
    print(n_reads, coverage, avg_size, max_size, avg_read_length)

    # define values for parameters to simulate reads
    genomes_id = [random.choice(genomes) for _ in range(n_reads)]
    start_positions = [random.choice(range(0, max_size, 1)) for _ in range(n_reads)]
    insert_sizes = [random.choice([400, 600, 1000]) for _ in range(n_reads)]
    strands = [random.choice(['fw', 'rev']) for _ in range(n_reads)]
    reads_lengths = [random.choice([100, 150, 250]) for _ in range(n_reads)]

    # define values of parameters for adding mutations
    # calculate number of mutations to add
    n_mut = math.ceil(n_reads * 250 * 0.5/100)
    print(n_mut)
    reads_indexes = [random.choice(range(0, n_reads-1, 1)) for _ in range(n_mut)]
    sites = [random.choice(range(0, 250, 1)) for _ in range(n_mut)]
    pairs = [random.choice(['fw', 'rev']) for _ in range(n_mut)]
    bases = [random.choice(['A', 'C', 'T', 'G']) for _ in range(n_mut)]
    # create dataframe
    df = pd.DataFrame(list(zip(reads_indexes, sites, pairs, bases)), columns=['indexes', 'sites', 'pairs', 'bases'])
    # sort dataframe by reads index
    df.sort_values(by=['indexes'], inplace=True)
    # reset rows numbers
    df.reset_index(drop=True, inplace=True)

    # count number of mutations added
    mut_count = 0
    with open(os.path.join(args.output_dir, f'{args.label}_{args.dataset}.fq'), 'w') as out_f:
        for i in range(n_reads):
            # concatenate 2 copies of the genome
            new_genome = fasta_seq[genomes_id[i]] + fasta_seq[genomes_id[i]]
            # define target sequence
            if strands[i] == 'fw':
                insert_seq = new_genome[start_positions[i]:start_positions[i]+insert_sizes[i]]
            elif strands[i] == 'rev':
                insert_seq = get_reverse(args, new_genome[start_positions[i]:start_positions[i]+insert_sizes[i]])
            # define forward and reverse reads
            forward_read = insert_seq[0:reads_lengths[i]]
            reverse_read = get_reverse(args, insert_seq[0:reads_lengths[i]], comp=True)

            # add mutations to the forward and/or reverse reads
            while mut_count < n_mut and df['indexes'][mut_count] == i:
                st = df['sites'].pop(mut_count)
                pr = df['pairs'].pop(mut_count)
                b = df['bases'].pop(mut_count)
                df['indexes'].pop(mut_count)
                # update site position in case it's above the read length
                if st >= reads_lengths[i]:
                    st = st % reads_lengths[i]
                if pr == 'fw':
                    forward_read = replace_base(forward_read, st, b)
                elif pr == 'rev':
                    reverse_read = replace_base(reverse_read, st, b)

                mut_count += 1

            # write pairs of reads to fastq file
            fw_read_id = f'@{genomes_id[i]}-label|{args.label}|-{i}/1'
            rv_read_id = f'@{genomes_id[i]}-label|{args.label}|-{i}/2'
            out_f.write(f'{fw_read_id}\n{forward_read}\n+\n{"?"*len(forward_read)}\n')
            out_f.write(f'{rv_read_id}\n{reverse_read}\n+\n{"?"*len(reverse_read)}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    parser.add_argument('--label', type=int, help='label of a species')
    parser.add_argument('--coverage', type=int, help='coverage used to estimate number of reads to simulate')
    parser.add_argument('--dataset', type=str, help='type of dataset', choices=['training', 'testing'])
    parser.add_argument('--fasta_dir', nargs='+', help='path to directories containing fasta files')
    args = parser.parse_args()

    # define base pairs
    args.base_pairs = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

    # get list of genomes associated with label
    genomes = get_genomes(args)
    # simulate paired-end reads
    simulate_reads(args, genomes)










