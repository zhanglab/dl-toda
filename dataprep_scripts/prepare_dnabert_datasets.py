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
import subprocess
# sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
# from dataprep_scripts.select_genomes import get_gtdb_info
# from select_genomes import get_gtdb_info

MIN_IDENTITY = 70
TICKS_INTERVAL = 500000
blastn_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/blastn"
makeblastdb_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/makeblastdb"
ncbi_datasets_exec = "/work/pi_yingzhang_uri_edu/ccres/tools/datasets"

def AddPctIdentity(dict_pident, sequences):
    up_sequences = []
    for i in range(len(sequences)):
        start = int(sequences[i].split('\t')[2])
        end = int(sequences[i].split('\t')[3])
        list_pident = []
        for j in range(start, end+1, 1):
            if j in dict_pident:
                list_pident.append(dict_pident[j])
            else:
                list_pident.append(0)
        avg_pident = sum(list_pident)/len(list_pident)
        up_sequences.append(sequences[i].rstrip() + f'\t{avg_pident}\n')
    return up_sequences

def GetMatchRegions(args, input_file, identity_thr=MIN_IDENTITY, key=None):
	# align_coords = []
    dict_pident = {}
    with open(input_file, 'r') as f:
        for count, line in enumerate(f, 1):
            sstart = int(line.rstrip().split(',')[2])
            send = int(line.rstrip().split(',')[3])
            qstart = int(line.rstrip().split(',')[4])
            qend = int(line.rstrip().split(',')[5])
            pident = float(line.rstrip().split(',')[8])
            qseq = line.rstrip().split(',')[9]
            sseq = line.rstrip().split(',')[10]
            if key == 'neg':
                for i in range(qstart, qend+1, 1):
                    dict_pident[i] = pident
            elif key == 'pos':
                for i in range(sstart, send+1, 1):
                    dict_pident[i] = pident

			# if pident >= identity_thr:
            # align_coords.append([qstart, qend, pident])

	# return align_coords, query_pident
    return dict_pident

def RunBlast(list_queries, list_labels, output_dir):
    for i in range(len(list_labels)):
        if not os.path.isdir(os.path.join(output_dir, list_labels[i])):
            os.makedirs(os.path.join(output_dir, list_labels[i]))
        # compare target genome with genome of negative label (query)
        result = subprocess.run([blastn_exec, '-query', f'{list_queries[i]}', '-task', 'blastn', '-db', f'{output_dir}/blastdb', '-out', f'{output_dir}/{list_labels[i]}/blastn.out', \
			 '-outfmt', "10 delim=, qseqid sseqid sstart send qstart qend qlen evalue pident qseq sseq sstrand", \
			 '-max_target_seqs', '5', '-num_threads', '1'])

def GetSequences(input_sam_data, input_cut_data, labels, sequences, bert_step, kmer):
    for i in range(len(labels)):
        with open(input_sam_data[i], 'r') as in_f:
            sequences[labels[i]] = [f'{labels[i]}\t' + s for s in in_f.readlines()]
        with open(input_cut_data[i], 'r') as in_f:
            sequences[labels[i]] += [f'{labels[i]}\t' + s for s in in_f.readlines()]


def GetNumberSequences(sequences, genome_size, min_coverage):
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

def GetTrainValData(args, sequences, out_f, label, train_genomes_df=None, label_train_size=None, label_val_size=None):
    seq_size = [len(s.rstrip().split('\t')[1].split(' ')) for s in sequences]
    # print(seq_size[0])
    # print(sequences[0].rstrip().split('\t')[1].split(' '))
    # print(len(sequences[0].rstrip().split('\t')[1].split(' ')))
    # print(f'{statistics.median(seq_size)}\t{min(seq_size)}\t{max(seq_size)}\t{statistics.mean(seq_size)}')
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
            train_size, val_size, sum_bases = GetNumberSequences(sequences, genome_size, args.min_coverage)
            out_f.write(f'{sum_bases/genome_size}\t')
        else:
            train_size = round(0.7*len(sequences))
            val_size = len(sequences) - train_size

    train_data = sequences[:train_size]
    val_data = sequences[-val_size:]

    out_f.write(f'{train_size}\t{val_size}\n')
    return train_data, val_data

def GetGenomeCov(data, train_genome_size, label, out_f, datatype): 
    print(train_genome_size)
    data_cov = {i: 0 for i in range(0, train_genome_size, 1)}
    print(len(data_cov))
    for i in range(len(data)):
        start = int(data[i].split('\t')[2])
        end = int(data[i].split('\t')[3])
        for j in range(start, end, 1):
            data_cov[j] += 1
    pct_genome_covered = (sum([1 for v in data_cov.values() if v != 0])/train_genome_size)*100
    coverage = sum(data_cov.values())/train_genome_size
    out_f.write(f'{datatype}\t{label}\t{pct_genome_covered}\t{coverage}\n')

def GetGenomeSize(list_fasta, list_labels):
    sizes = {}
    for i in range(len(list_labels)):
        seq = ''
        with open(list_fasta[i], 'r') as f:
            for line in f:
                if line[0] != '>':
                    seq += line.rstrip()
        sizes[list_labels[i]] = len(seq)
    return sizes


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
    parser.add_argument('--pos_label', type=str, help='labels of species acting as the positive class for a binary classifier')
    parser.add_argument('--neg_label', type=str, nargs='+', help='labels of species acting as the negative class for a binary classifier')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # if args.bert_step == 'pretraining' or args.multiclass:
    if args.dataset == 'train':
        # get training genomes
        train_genomes_df = pd.read_csv(args.train_genomes_info, header=None, sep="\t")
        train_genomes_df.columns = ['label','genome','fasta']
        # get size of training genomes
        pos_train_genome = train_genomes_df[train_genomes_df['label'] == int(args.pos_label)]['genome'].tolist()[0]
        pos_train_fasta = train_genomes_df[train_genomes_df['label'] == int(args.pos_label)]['fasta'].tolist()[0]
        neg_train_fasta = train_genomes_df[train_genomes_df['label'].isin([int(i) for i in args.neg_label])]['fasta'].tolist()
        neg_train_genomes = train_genomes_df[train_genomes_df['label'].isin([int(i) for i in args.neg_label])]['genome'].tolist()
        print(neg_train_fasta)
        print(neg_train_genomes)
        print(args.neg_label)
        genomes_size = GetGenomeSize(neg_train_fasta+[pos_train_fasta], args.neg_label+[args.pos_label])
        print(genomes_size)

    
    labels = args.neg_label + [args.pos_label]
    input_sam_data = [i for i in sorted(glob.glob(f"{args.input_dir}/{args.dataset}_data_label_*/k{args.kmer}/data_sam_*_k{args.kmer}")) if 'seq' not in i.rstrip().split('/')[-1] and i.rstrip().split('/')[-3].split('_')[3] in labels]
    input_cut_data = [i for i in sorted(glob.glob(f"{args.input_dir}/{args.dataset}_data_label_*/k{args.kmer}/data_cut_*_k{args.kmer}")) if 'seq' not in i.rstrip().split('/')[-1] and i.rstrip().split('/')[-3].split('_')[3] in labels]
    
    assert len(input_sam_data) == len(input_cut_data), f"Missing {args.dataset} dnabert data"

    chunk_size = math.ceil(len(labels)/args.num_processes) if len(labels) > args.num_processes else 1
    grouped_labels = [labels[i:i+chunk_size] for i in range(0, len(labels), chunk_size)]
    grouped_sam_data = [input_sam_data[i:i+chunk_size] for i in range(0, len(input_sam_data), chunk_size)]
    grouped_cut_data = [input_cut_data[i:i+chunk_size] for i in range(0, len(input_cut_data), chunk_size)]
    grouped_neg_labels = [args.neg_label[i:i+chunk_size] for i in range(0, len(args.neg_label), chunk_size)]
    grouped_fasta = [neg_train_fasta[i:i+chunk_size] for i in range(0, len(neg_train_fasta), chunk_size)]
    print(grouped_neg_labels)
    print(grouped_fasta)
    # create BLAST database for target genome (genome of positive label)
    blastoutdir = os.path.join(args.output_dir, 'blast', pos_train_genome)
    if not os.path.isdir(blastoutdir):
        os.makedirs(blastoutdir)
    result = subprocess.run([makeblastdb_exec, '-in', f'{pos_train_fasta}', '-input_type', 'fasta', '-dbtype', 'nucl', '-out', f'{blastoutdir}/blastdb'])
    
    with mp.Manager() as manager: 
        # blast genomes
        processes = [mp.Process(target=RunBlast, args=(grouped_fasta[i], grouped_neg_labels[i], blastoutdir)) for i in range(len(grouped_neg_labels))]
        for p in processes:
            p.start() 
        for p in processes:
            p.join() 
        
        # get sequences
        sequences = manager.dict()
        processes = [mp.Process(target=GetSequences, args=(grouped_sam_data[i], grouped_cut_data[i], grouped_labels[i], sequences, args.bert_step, args.kmer)) for i in range(len(grouped_labels))]
        for p in processes:
            p.start() 
        for p in processes:
            p.join() 

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

                label_train_size, label_val_size, _ = GetNumberSequences(sequences[str(largest_genome_label)], largest_genome_size, args.min_coverage)

                print(f'largest genome train size: {label_train_size}\tlargest genome val size: {label_val_size}')

            all_train_data = []
            all_val_data = []
            with open(os.path.join(args.output_dir, f'data_info_k{args.kmer}.tsv'), 'w') as out_f:
                for l in labels:
                    GetTrainValData(args, sequences[l], all_train_data, all_val_data, out_f, label=l, train_genomes_df=train_genomes_df, label_train_size=label_train_size, label_val_size=label_val_size)
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
                # calculate the number of sequences to sample
                num = len(sequences[args.pos_label])
                # get sequences
                all_train_data = []
                all_val_data = []
                with open(os.path.join(args.output_dir, f'{args.bert_step}_l{args.pos_label}_train_data_info_k{args.kmer}.tsv'), 'w') as out_f:
                    # for other species
                    labels_other = [l for l in labels if l != args.pos_label]
                    print(f'get sequences from label 0\t# species: {len(labels_other)}')
                    num_sp_labels = len(labels_other)
                    if num_sp_labels > 1:
                        num_seq_per_sp = [num // num_sp_labels + (1 if x < num % num_sp_labels else 0) for x in range (num_sp_labels)]
                    else:
                        num_seq_per_sp = [num]
                    for i in range(len(labels_other)):
                        # get results from alignment with train genome of positive label
                        dict_pident = GetMatchRegions(args, os.path.join(blastoutdir, labels_other[i], 'blastn.out'), identity_thr=MIN_IDENTITY, key='neg')
                        seq = sequences[labels_other[i]]
                        random.shuffle(seq)
                        num_seq = num_seq_per_sp.pop()
                        label_seq = seq[:num_seq]
                        label_seq = AddPctIdentity(dict_pident, label_seq)
                        train_data, val_data = GetTrainValData(args, label_seq, out_f, labels_other[i])
                        GetGenomeCov(train_data, genomes_size[labels_other[i]], labels_other[i], out_f, 'train')
                        GetGenomeCov(val_data, genomes_size[labels_other[i]], labels_other[i], out_f, 'val')
                        all_train_data += train_data
                        all_val_data += val_data
                    # split sequences between train and val datasets
                    # update sequences with average percentage identity with negative genome
                    pos_label_seq = AddPctIdentity(dict_pident, sequences[args.pos_label])
                    print('split sequences between train and val datasets for label 1')
                    train_data, val_data = GetTrainValData(args, pos_label_seq, out_f, args.pos_label)
                    # calculate percentage of training genome covered in train and val datasets
                    GetGenomeCov(train_data, genomes_size[args.pos_label], args.pos_label, out_f, 'train')
                    GetGenomeCov(val_data, genomes_size[args.pos_label], args.pos_label, out_f, 'val')
                    all_train_data += train_data
                    all_val_data += val_data

                random.shuffle(all_val_data)
                random.shuffle(all_train_data)

                with open(os.path.join(args.output_dir, f'{args.bert_step}_l{args.pos_label}_train_data_k{args.kmer}.tsv'), 'w') as out_f:
                    out_f.write(''.join(all_train_data))

                with open(os.path.join(args.output_dir, f'{args.bert_step}_l{args.pos_label}_val_data_k{args.kmer}.tsv'), 'w') as out_f:
                    out_f.write(''.join(all_val_data))

            elif args.dataset == 'test':
                # only for finetuning
                out_info = open(os.path.join(args.output_dir, f'{args.bert_step}_l{args.pos_label}_test_data_info_k{args.kmer}.tsv'), 'w')
                labels_other = [l for l in labels if l != args.pos_label]
                print(f'# other labels: {len(labels_other)}')
                num_sp_labels = len(labels_other)
                num = len(sequences[args.pos_label])
                num_seq_per_sp = [num // num_sp_labels + (1 if x < num % num_sp_labels else 0) for x in range (num_sp_labels)]
                print(num, num_sp_labels, len(num_seq_per_sp), num_seq_per_sp[:3])
                with open(os.path.join(args.output_dir, f'{args.bert_step}_l{args.pos_label}_test_data_k{args.kmer}.tsv'), 'w') as out_f:
                    for i in range(len(labels)):
                        if labels[i] == args.pos_label:
                            out_f.write(''.join(sequences[labels[i]]))
                            out_info.write(f'{labels[i]}\t{len(sequences[labels[i]])}\n')
                        else:
                            num_seq = num_seq_per_sp.pop()
                            l_sequences = sequences[labels[i]][:num_seq]
                            out_f.write(''.join(l_sequences))
                            out_info.write(f'{labels[i]}\t{len(l_sequences)}\n')


if __name__ == "__main__":
    main()
