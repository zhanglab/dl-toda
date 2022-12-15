import sys
import os
import random
import math
import json
from collections import defaultdict
import glob
import multiprocessing as mp

def create_sets(reads, set_type, taxa2labels, output_dir):
    # calculate number of sets of reads
    num_reads = 0
    for process, process_reads in reads.items():
        num_reads += process_reads

    num_sets = math.ceil(num_reads/20000000) if num_reads > 20000000 else 1

    genome_out = open(os.path.join(output_dir, f'{set_type}-genome-read-count'), 'w')
    label_out = open(os.path.join(output_dir, f'{set_type}-label-read-count'), 'w')
    for label in taxa2labels.values():
        label_fq_files = sorted(glob.glob(os.path.join(output_dir, set_type, 'reads-*' , f'{set_type}-*-{label}.fq')))
        list_reads = []
        for fastq in label_fq_files:
            with open(fastq, 'r') as f:
                genome = fastq.rstrip().split('/')[-1].split('-')[1]
                content = f.readlines()
                reads = [''.join(content[j:j+4]) for j in range(0, len(content), 4)]
                list_reads += reads
                genome_out.write(f'{genome}\t{len(reads)}\n')
        random.shuffle(list_reads)
        num_reads_per_set = math.ceil(len(list_reads)/num_sets)
        label_out.write(f'{label}\t{len(list_reads)}\t{num_reads_per_set}\n')
        for count, i in enumerate(range(0, len(list_reads), num_reads_per_set)):
            with open(os.path.join(output_dir, f'{set_type}-subset-{count}.fq'), 'a') as outfile:
                outfile.write(''.join(list_reads[i:i+num_reads_per_set]))


def split_reads(grouped_files, output_dir, genomes2labels, taxa2labels, process_id, train_reads, val_reads):

    # create directories to store output fq files
    process_train_reads = 0
    process_val_reads = 0
    if not os.path.exists(os.path.join(output_dir, 'training', f'reads-{process_id}')):
        os.makedirs(os.path.join(output_dir, 'training', f'reads-{process_id}'))
    if not os.path.exists(os.path.join(output_dir, 'validation', f'reads-{process_id}')):
        os.makedirs(os.path.join(output_dir, 'validation', f'reads-{process_id}'))

    for fq_file in grouped_files:
        genome = fq_file.rstrip().split('/')[-1][:-4]
        label = genomes2labels[genome]
        with open(fq_file, 'r') as f:
            content = f.readlines()
            reads = [''.join(content[j:j+4]) for j in range(0, len(content), 4)]
            random.shuffle(reads)
            num_train_reads = math.ceil(0.7*len(reads))
            with open(os.path.join(output_dir, 'training', f'reads-{process_id}', f'train-{genome}-{label}.fq'), 'w') as out_f:
                out_f.write(''.join(reads[:num_train_reads]))
            with open(os.path.join(output_dir, 'validation', f'reads-{process_id}', f'val-{genome}-{label}.fq'), 'w') as out_f:
                out_f.write(''.join(reads[num_train_reads:]))
            process_train_reads += num_train_reads
            process_val_reads += len(reads) - num_train_reads

    train_reads[process_id] = process_train_reads
    val_reads[process_id] = process_val_reads


def create_train_val_sets(input_dir, output_dir, genomes2labels, taxa2labels):
    # get list of fastq files in input directory
    fq_files = glob.glob(os.path.join(input_dir, '*.fq'))
    chunk_size = math.ceil(len(fq_files)/mp.cpu_count())
    grouped_files = [fq_files[i:i+chunk_size] for i in range(0,len(fq_files),chunk_size)]

    with mp.Manager() as manager:
        train_reads = manager.dict()
        val_reads = manager.dict()
        processes = [mp.Process(target=split_reads, args=(grouped_files[i], output_dir, genomes2labels, taxa2labels, i, train_reads, val_reads)) for i in range(len(grouped_files))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        create_sets(train_reads, 'training', taxa2labels, output_dir)
        create_sets(val_reads, 'validation', taxa2labels, output_dir)


def main():
    input_dir = sys.argv[1] # directory containing fastq files of simulated reads
    output_dir = sys.argv[2]
    file_w_genomes = sys.argv[3] # tab separated file with genome and their label
    labels_file = sys.argv[4] # path to json file mapping labels to taxa in dl-toda

    ranks = {'species': 0, 'genus': 1, 'family': 2, 'order': 3, 'class': 4, 'phylum': 5}

    with open(labels_file, 'r') as f:
        labels2taxa = json.load(f)
        taxa2labels = {v:k for k, v in labels2taxa.items()}

    with open(file_w_genomes) as f:
        content = f.readlines()
        genomes2labels = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[1] for line in content}

    if not os.path.exists(os.path.join(output_dir, 'training')):
        os.makedirs(os.path.join(output_dir, 'training'))
    if not os.path.exists(os.path.join(output_dir, 'validation')):
        os.makedirs(os.path.join(output_dir, 'validation'))
    create_train_val_sets(input_dir, output_dir, genomes2labels, taxa2labels)



if __name__ == "__main__":
    main()
