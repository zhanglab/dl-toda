import glob
import sys
import random
import multiprocessing as mp
import os
import math
import argparse
import pandas as pd
import statistics
from Bio import SeqIO
# sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
# from dataprep_scripts.select_genomes import get_gtdb_info
# from select_genomes import get_gtdb_info

def get_sequences(input_sam_data, input_cut_data, labels, sequences, bert_step, kmer):
    for i in range(len(labels)):
        with open(input_sam_data[i], 'r') as in_f:
            sequences[labels[i]] = [f'{labels[i]}\t' + s for s in in_f.readlines()]
        with open(input_cut_data[i], 'r') as in_f:
            sequences[labels[i]] += [f'{labels[i]}\t' + s for s in in_f.readlines()]


def get_number_sequences(sequences, genome_size, min_coverage):
    # get number of sequences to have at least 1x coverage for training 
    sum_bases = 0
    num_seqs = 0
    for s in sequences:
        seq = s.rstrip().split('\t')[1].split(" ")
        original_seq = seq[0]
        for i in range(1, len(seq), 1):
            original_seq += seq[i][-1]
        sum_bases += len(original_seq)
        num_seqs += 1
        if sum_bases / genome_size >= min_coverage:
            break
    
    train_size = round(0.7*num_seqs)
    val_size = num_seqs - train_size

    return train_size, val_size, sum_bases

def get_train_val_data(args, sequences, all_train_data, all_val_data, out_f, label=None, train_genomes_df=None, label_train_size=None, label_val_size=None):
    seq_size = [len(s.rstrip().split('\t')[1].split(' ')) for s in sequences]
    # print(seq_size[0])
    # print(sequences[0].rstrip().split('\t')[1].split(' '))
    # print(len(sequences[0].rstrip().split('\t')[1].split(' ')))
    print(f'{statistics.median(seq_size)}\t{min(seq_size)}\t{max(seq_size)}\t{statistics.mean(seq_size)}')
    random.shuffle(sequences)
    out_f.write(f'{label}\t')
    if args.multiclass:
        if len(sequences) < label_train_size + label_val_size:
            num_extra_seqs = label_train_size + label_val_size - len(sequences)
            add_seqs_indices = [random.randint(0, len(sequences)-1) for i in range(num_extra_seqs)]
            add_seqs = []
            for idx in add_seqs_indices:
                add_seqs.append(sequences[idx])
            sequences += add_seqs
        
        train_size = label_train_size
        val_size = label_val_size
    else:
        if args.bert_step == 'pretraining':
            fasta = train_genomes_df[train_genomes_df[0]==int(label)][2].tolist()[0]
            genome_size = get_genome_size(fasta)
            train_size, val_size, sum_bases = get_number_sequences(sequences, genome_size, args.min_coverage)
            out_f.write(f'{sum_bases/genome_size}\t')
        else:
            train_size = round(0.7*len(sequences))
            val_size = len(sequences) - train_size 

    all_train_data += sequences[:train_size]
    all_val_data += sequences[-val_size:]

    out_f.write(f'{train_size}\t{val_size}\n')

def GetGenomeCov(data, train_genome_size):
    data_cov = {i: 0 for i in range(train_genome_size)}
    for i in range(len(data)):
        start = int(data[i].split('\t')[2])
        end = int(data[i].split('\t')[3])
        for j in range(start, end+1, 1):
            data_cov[j] += 1
        print(start, end, end+1, j)
    pct_genome_covered = (sum([1 for k, v in range(len(data_cov)) if v != 0])/train_genome_size)*100
    print(sum([1 for k, v in range(len(data_cov)) if v != 0]), train_genome_size)
    return pct_genome_covered


def get_genome_size(fasta):
    seq = ''
    with open(fasta, 'r') as f:
        for line in f:
            if line[0] != '>':
                seq += line.rstrip()
    return len(seq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='path to input directory')
    parser.add_argument('--output_dir', type=str, help='path to output file')
    # parser.add_argument('--gtdb_info', type=str, help='path to bac120_metadata_r220.tsv file')
    parser.add_argument('--train_genomes_info', type=str, help='path to train_genomes.tsv file')
    parser.add_argument('--dataset', type=str, help='type of dataset to prepare', choices=['train', 'test'])
    parser.add_argument('--bert_step', choices=['pretraining', 'finetuning'])
    parser.add_argument('--kmer', type=int, help='length of kmers')
    parser.add_argument('--min_coverage', type=float, help='minimun coverage of training genome', default=1.5)
    parser.add_argument('--multiclass', action='store_true', default=False)
    parser.add_argument('--num_processes', type=int, help='number of processes to run in parallel')
    parser.add_argument('--target_label', type=str, help='positive class for binary classifiers')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # if args.bert_step == 'pretraining' or args.multiclass:
    # get size of training genomes
    train_genomes_df = pd.read_csv(args.train_genomes_info, header=None, sep="\t")
    train_genomes_df.columns = ['label','genome','fasta']
    print(train_genomes_df)
    # get size of training genome
    train_fasta = train_genomes_df[train_genomes_df['label'] == int(args.target_label)]['fasta']
    print(train_fasta)
    for seq_record in SeqIO.parse(train_fasta, "fasta"):
        print('genome size', len(seq_record.seq))
    train_genome_size = get_genome_size(train_fasta)
    print('genome size', train_genome_size)
    sys.exit(1)

    input_sam_data = [i for i in sorted(glob.glob(f"{args.input_dir}/{args.dataset}_data_label_*/k{args.kmer}/data_sam_*_k{args.kmer}")) if 'seq' not in i.rstrip().split('/')[-1]]
    input_cut_data = [i for i in sorted(glob.glob(f"{args.input_dir}/{args.dataset}_data_label_*/k{args.kmer}/data_cut_*_k{args.kmer}")) if 'seq' not in i.rstrip().split('/')[-1]]
    sam_labels = [i.rstrip().split('/')[-3].split('_')[3] for i in input_sam_data]
    cut_labels = [i.rstrip().split('/')[-3].split('_')[3] for i in input_cut_data]
    print(len(input_sam_data), len(input_cut_data))
    
    if cut_labels != sam_labels:
        raise Exception(f"Missing {args.dataset} dnabert data for label {set(sam_labels).difference(cut_labels)}")
    else:
        labels = sam_labels

    # labels = ['36', '40']
    # input_sam_data = sorted(glob.glob(f"{args.input_dir}/*/{args.dataset}_data_label_36/pretraining/data_sam_k{args.kmer}_cleaned.tsv"))
    # input_cut_data = sorted(glob.glob(f"{args.input_dir}/*/{args.dataset}_data_label_36/pretraining/data_cut_k{args.kmer}_cleaned.tsv"))

    # input_sam_data += glob.glob(f"{args.input_dir}/*/{args.dataset}_data_label_0/pretraining/data_sam_k{args.kmer}_cleaned.tsv")
    # input_cut_data += glob.glob(f"{args.input_dir}/*/{args.dataset}_data_label_0/pretraining/data_cut_k{args.kmer}_cleaned.tsv")

    chunk_size = math.ceil(len(labels)/args.num_processes)
    grouped_labels = [labels[i:i+chunk_size] for i in range(0, len(labels), chunk_size)]
    grouped_sam_data = [input_sam_data[i:i+chunk_size] for i in range(0, len(input_sam_data), chunk_size)]
    grouped_cut_data = [input_cut_data[i:i+chunk_size] for i in range(0, len(input_cut_data), chunk_size)]

    with mp.Manager() as manager: # create manager object to allow processes to manipulate python data structures
        sequences = manager.dict()
        # create list of Process objects
        processes = [mp.Process(target=get_sequences, args=(grouped_sam_data[i], grouped_cut_data[i], grouped_labels[i], sequences, args.bert_step, args.kmer)) for i in range(args.num_processes)]
        for p in processes:
            p.start() # start the processes
        for p in processes:
            p.join() # join the processes, program will hang and wait until all the processes are done

        if args.bert_step == 'pretraining' or args.multiclass:
            # args.min_coverage == 1.5 for pre-training
            if args.multiclass:
                # get largest genome in training dataset
                genomes_size = {}
                for i in range(len(train_genomes_df)):
                    genomes_size[train_genomes_df[0][i]] = get_genome_size(train_genomes_df[2][i])
                
                # get number of sequences of largest genome in the dataset with a coverage of 1x
                largest_genome_label = max(genomes_size, key=genomes_size.get)
                largest_genome_size = max(genomes_size.values())
                print(f'largest genome size: {largest_genome_size}\tlabel: {largest_genome_label}')

                label_train_size, label_val_size, _ = get_number_sequences(sequences[str(largest_genome_label)], largest_genome_size, args.min_coverage)

                print(f'largest genome train size: {label_train_size}\tlargest genome val size: {label_val_size}')

            all_train_data = []
            all_val_data = []
            with open(os.path.join(args.output_dir, f'data_info_k{args.kmer}.tsv'), 'w') as out_f:
                for l in labels:
                    get_train_val_data(args, sequences[l], all_train_data, all_val_data, out_f, label=l, train_genomes_df=train_genomes_df, label_train_size=label_train_size, label_val_size=label_val_size)
                out_f.write(f'total\t{len(all_train_data)}\t{len(all_val_data)}')

            random.shuffle(all_val_data)
            random.shuffle(all_train_data)

            with open(os.path.join(args.output_dir, f'train_data_k{args.kmer}.tsv'), 'w') as out_f:
                out_f.write(''.join(all_train_data))

            with open(os.path.join(args.output_dir, f'val_data_k{args.kmer}.tsv'), 'w') as out_f:
                out_f.write(''.join(all_val_data))

        elif args.bert_step == "finetuning":
            # # load gtdb metadata
            # genomes, _, _, _, _, gtdb_taxonomy = get_gtdb_info(args.gtdb_info)
            # genome_to_tax = dict(zip(genomes, gtdb_taxonomy))
            if args.dataset == 'train':
                # train_genomes_df = pd.read_csv(args.train_genomes_info, header=None, sep="\t")
                # train_genomes_df.columns = ['label','genome','fasta']
                # # get training genome of target label
                # target_genome = train_genomes_df.loc[train_genomes_df['label'] == int(args.target_label), 'genome'].tolist()[0]
                # print(target_genome)
                # # get genus of target label
                # target_genus = genome_to_tax[target_genome].split(';')[-2].split('__')[1]
                # # get labels with same genus
                # labels_same_genus = []
                # train_genomes = train_genomes_df['genome'].tolist()
                # train_labels = train_genomes_df['label'].tolist()
                # for i in range(len(train_genomes)):
                #     if train_genomes[i]!= target_genome and train_genomes[i] in genome_to_tax:
                #         if target_genus in genome_to_tax[train_genomes[i]].split(';')[-2].split('__')[1]:
                #             labels_same_genus.append(str(train_labels[i]))
                # print(labels_same_genus, len(labels_same_genus))
                # calculate the number of sequences to sample
                num = len(sequences[args.target_label])
                # num_genus_labels = len(labels_same_genus)
                # num_seq_per_genus = [num // num_genus_labels + (1 if x < num % num_genus_labels else 0) for x in range (num_genus_labels)]
                                
                # get sequences
                other_labels_seq = []
                all_train_data = []
                all_val_data = []
                with open(os.path.join(args.output_dir, f'{args.bert_step}_l{args.target_label}_train_data_info_k{args.kmer}.tsv'), 'w') as out_f:
                    # # at the genus level
                    # for i in range(len(labels_same_genus)):
                    #     num_seq = num_seq_per_genus.pop()
                    #     seq = sequences[labels_same_genus[i]]
                    #     random.shuffle(seq)
                    #     # other_labels_seq += sequences[labels_same_genus[i]][:num_seq]
                    #     other_labels_seq += seq[:num_seq]
                    # print(f'# sequences: {len(other_labels_seq)}')
                    
                    # for other species
                    # if num != len(sequences[args.target_label]):
                    # labels_other = [l for l in labels if l not in labels_same_genus and l != args.target_label]
                    labels_other = [l for l in labels if l != args.target_label]
                    print(f'# other labels: {len(labels_other)}')
                    num_sp_labels = len(labels_other)
                    num_seq_per_sp = [num // num_sp_labels + (1 if x < num % num_sp_labels else 0) for x in range (num_sp_labels)]
                    for i in range(len(labels_other)):
                        seq = sequences[labels_other[i]]
                        random.shuffle(seq)
                        num_seq = num_seq_per_sp.pop()
                        other_labels_seq += seq[:num_seq]
                        # other_labels_seq += sequences[labels_other[i]][:num_seq]
                    print(f'# sequences: {len(other_labels_seq)}')
                    
                    # split sequences between train and val datasets
                    print('split sequences between train and val datasets for label 1')
                    get_train_val_data(args, sequences[args.target_label], all_train_data, all_val_data, out_f, label=args.target_label)
                    # calculate percentage of training genome covered in train and val datasets
                    train_pct_genome_covered = GetGenomeCov(all_train_data, train_genome_size)
                    out_f.write(f'% train genome covered in train dataset\t{train_pct_genome_covered}')
                    val_pct_genome_covered = GetGenomeCov(all_val_data, train_genome_size)
                    out_f.write(f'% train genome covered in val dataset\t{_pct_genome_covered}')
                    print('split sequences between train and val datasets for label 0')
                    get_train_val_data(args, other_labels_seq, all_train_data, all_val_data, out_f, label='other labels')
                    

                random.shuffle(all_val_data)
                random.shuffle(all_train_data)

                with open(os.path.join(args.output_dir, f'{args.bert_step}_l{args.target_label}_train_data_k{args.kmer}.tsv'), 'w') as out_f:
                    out_f.write(''.join(all_train_data))

                with open(os.path.join(args.output_dir, f'{args.bert_step}_l{args.target_label}_val_data_k{args.kmer}.tsv'), 'w') as out_f:
                    out_f.write(''.join(all_val_data))

            elif args.dataset == 'test':
                # only for finetuning
                out_info = open(os.path.join(args.output_dir, f'{args.bert_step}_l{args.target_label}_test_data_info_k{args.kmer}.tsv'), 'w')
                labels_other = [l for l in labels if l != args.target_label]
                print(f'# other labels: {len(labels_other)}')
                num_sp_labels = len(labels_other)
                num = len(sequences[args.target_label])
                num_seq_per_sp = [num // num_sp_labels + (1 if x < num % num_sp_labels else 0) for x in range (num_sp_labels)]
                print(num, num_sp_labels, len(num_seq_per_sp), num_seq_per_sp[:3])
                with open(os.path.join(args.output_dir, f'{args.bert_step}_l{args.target_label}_test_data_k{args.kmer}.tsv'), 'w') as out_f:
                    for i in range(len(labels)):
                        if labels[i] == args.target_label:
                            out_f.write(''.join(sequences[labels[i]]))
                            out_info.write(f'{labels[i]}\t{len(sequences[labels[i]])}\n')
                        else:
                            num_seq = num_seq_per_sp.pop()
                            l_sequences = sequences[labels[i]][:num_seq]
                            out_f.write(''.join(l_sequences))
                            out_info.write(f'{labels[i]}\t{len(l_sequences)}\n')


if __name__ == "__main__":
    main()
