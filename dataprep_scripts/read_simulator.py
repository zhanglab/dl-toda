import os
import sys
import glob
import argparse
import math
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
            fasta_seq['_'.join(fa.split('_')[2:4])] = compute_size(fa)
    # compute average size of selected genomes
    avg_size = sum([len(fasta_seq[g]) for g in genomes])/len(genomes)
    # get the largest size of all genomes
    max_size = max([len(fasta_seq[g]) for g in genomes])
    return fasta_seq, avg_size, max_size


def get_genomes(args):
    file_w_genomes = '/'.join(
        os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + f'/data/{args.dataset}_genomes.tsv'
    with open(file_w_genomes, 'r') as f:
        for line in f:
            if line.rstrip().split('\t')[1] == str(args.label):
                genomes.append(line.rstrip().split('\t')[0])
    return genomes


def get_reverse(args, sequence, comp=False):
    rev_seq = ''.join([args.base_pairs[b] if b in args.base_pairs else b for b in sequence])
    if comp:
        # return reverse complement sequence (negative/lower/reverse strand in the 5' to 3' end direction)
        return rev_seq[::-1]
    else:
        # return reverse sequence (negative/lower/reverse strand in the 3' to 5' end direction)
        return rev_seq


def simulate_reads(args, genomes):
    # compute average and max genome size
    fasta_seq, avg_size, max_size = get_genomes_size(args, genomes)
    # compute number of reads to simulate
    coverage = 7 if args.dataset == 'training' else 3
    avg_read_length = sum([100, 150, 250])/3
    # get number of pairs of reads to simulate
    n_reads = math.ceil(coverage * avg_size /(avg_read_length*2))
    print(n_reads)

    # define values for parameters to simulate reads
    genomes_id = [random.choice(genomes) for _ in range(n_reads)]
    start_positions = [random.choice(range(0, max_size, 1)) for _ in range(n_reads)]
    insert_sizes = [random.choice([400, 600, 1000]) for _ in range(n_reads)]
    strands = [random.choice(['fw', 'rev']) for _ in range(n_reads)]
    read_lengths = [random.choice([100, 150, 250]) for _ in range(n_reads)]

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
    df.sort_values(by=['indexes'])

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
            forward_read = insert_seq[0:read_lengths[i]]
            reverse_read = get_reverse(args, insert_seq[0:read_lengths[i]], comp=True)

            # add mutations to the forward and reverse reads
            while df['indexes'][0] == i:
                st = df['sites'].pop(0)
                pr = df['pairs'].pop(0)
                b = df['bases'].pop(0)
                # update site position in case it's above the read length
                if st >= reads_lengths[i]:
                    st = st % read_lengths[i]
                if pr == 'fw':
                    forward_read[st] = b
                elif pr == 'rev':
                    reverse_read[st] = b
                fw_read_id = f'@{genome_id[i]}-label|{args.label}|-{i}/1'
                rv_read_id = f'@{genome_id[i]}-label|{args.label}|-{i}/2'
                out_f.write(f'{fw_read_id}\n{forward_read}\n+\n{"?"*len(forward_read)}\n')
                out_f.write(f'{rv_read_id}\n{reverse_read}\n+\n{"?"*len(reverse_read)}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    parser.add_argument('--label', type=int, help='label of a species')
    parser.add_argument('--dataset', type=str, help='type of dataset', choices=['training', 'testing'])
    parser.add_argument('--fasta_dir', nargs='+', help='path to directories containing fasta files')
    args = parser.parse_args()

    # define base pairs
    args.base_pairs = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

    # get list of genomes associated with label
    genomes = get_genomes(args)
    # simulate paired-end reads
    simulate_reads(args, genomes)










