import os
import sys
import tensorflow as tf
import numpy as np
import random
import datetime
from collections import defaultdict
# from Bio import SeqIO
import argparse
import json
import glob
import math
import gzip
import multiprocessing as mp
from tfrecords_utils import vocab_dict, get_kmer_arr, prepare_input_data
from tfrecords_bert_utils import *


def wrap_read(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def wrap_label(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_weights(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_meta_tfrecords(args, grouped_files):
    for fq_file in grouped_files:
        """ Converts metagenomic reads to tfrecords """
        output_prefix = '.'.join(fq_file.split('/')[-1].split('.')[0:-2]) if fq_file[-2:] == 'gz' else '.'.join(fq_file.split('/')[-1].split('.')[0:-1])
        output_tfrec = os.path.join(args.output_dir, output_prefix + '.tfrec')
        outfile = open('/'.join([args.output_dir, output_prefix + f'-read_ids.tsv']), 'w')
        with tf.compat.v1.python_io.TFRecordWriter(output_tfrec) as writer:
            if fq_file[-2:] == 'gz':
                handle = gzip.open(fq_file, 'rt')
            else:
                handle = open(fq_file, 'r')
            # with gzip.open(args.input_fastq, 'rt') as handle:
            content = handle.readlines()
            reads = [''.join(content[j:j+4]) for j in range(0, len(content), 4)]
            for count, rec in enumerate(reads, 1):
            # for count, rec in enumerate(SeqIO.parse(handle, 'fastq'), 1):
            #     read = str(rec.seq)
                read = rec.split('\n')[1].rstrip()
                # read_id = rec.description
                read_id = rec.split('\n')[0].rstrip()
                # outfile.write(f'{read_id}\t{count}\n')
                kmer_array = get_kmer_arr(args, read)
                data = \
                    {
                        'read': wrap_read(kmer_array),
                        'label': wrap_label(count)
                    }
                feature = tf.train.Features(feature=data)
                example = tf.train.Example(features=feature)
                serialized = example.SerializeToString()
                writer.write(serialized)

            with open(os.path.join(args.output_dir, output_prefix + '-read_count'), 'w') as f:
                f.write(f'{count}')

        outfile.close()


# def get_data_for_bert(args, nsp_data, data, list_reads, grouped_reads, grouped_reads_index, process):
#     process_data = []
#     process_nsp_data = {}
#     for i, r in enumerate(grouped_reads):
#         label = int(r.rstrip().split('\n')[0].split('|')[1])
#         if args.update_labels:
#             label = int(args.labels_mapping[str(label)])
#         # update sequence
#         segment_1, segment_2, nsp_label = split_read(args, list_reads, r.rstrip().split('\n')[1], grouped_reads_index[i], process)
#         # parse dna sequences
#         segment_1_list = get_kmer_arr(args, segment_1, args.read_length//2, args.kmer_vector_length)
#         segment_2_list = get_kmer_arr(args, segment_2, args.read_length//2, args.kmer_vector_length)
#         # prepare input for next sentence prediction task
#         dna_list, segment_ids = get_nsp_input(args, segment_1_list, segment_2_list)
#         # mask 15% of k-mers in reads
#         if args.bert_step == 'pretraining':
#             input_ids, input_mask, masked_lm_weights, masked_lm_positions, masked_lm_ids = get_mlm_input(args, dna_list)
#             process_data.append([input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_weights, masked_lm_ids, nsp_label, label])
#         elif args.bert_step == 'finetuning':
#             # create input_mask vector indicating padded values
#             input_mask = [1] * len(dna_list)
#             process_data.append([dna_list, input_mask, segment_ids, label])
#             print(dna_list, input_mask, segment_ids, label)
#         if label not in process_nsp_data:
#             process_nsp_data[label] = defaultdict(int)
#             process_nsp_data[label][str(nsp_label)] += 1
#         else:
#             process_nsp_data[label][str(nsp_label)] += 1

#     data[process] = process_data
#     nsp_data[process] = process_nsp_data


def get_data_for_bert_1(args, dna_sequences, labels, reads_index):
    """ process data obtained with ART read simulator"""
    data = []
    nsp_data = {}
    for i in range(len(dna_sequences)):
        label = labels[i]

        if args.update_labels:
            label = int(args.labels_mapping[str(label)])
 
        # get input for nsp task
        segment_1, segment_2, nsp_label = split_read(args, dna_sequences, dna_sequences[i], i)
 
        # parse dna sequences
        segment_1_list = get_kmer_arr(args, segment_1, args.read_length//2, args.kmer_vector_length)
        segment_2_list = get_kmer_arr(args, segment_2, args.read_length//2, args.kmer_vector_length)
        # prepare input for next sentence prediction task
        dna_list, segment_ids = get_nsp_input(args, segment_1_list, segment_2_list)
        # mask 15% of k-mers in reads
        if args.bert_step == 'pretraining':
            input_ids, input_mask, masked_lm_weights, masked_lm_positions, masked_lm_ids = get_mlm_input(args, dna_list)
            data.append([input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_weights, masked_lm_ids, nsp_label])
        elif args.bert_step == 'finetuning':
            # create input_mask vector indicating padded values. Padding token indices are masked (0) to avoid
            # performing attention on them.
            input_mask = [1] * len(dna_list)
            data.append([dna_list, input_mask, segment_ids, label])
            # print(r, dna_list, input_mask, segment_ids, label)
        if label not in nsp_data:
            nsp_data[label] = defaultdict(int)
            nsp_data[label][str(nsp_label)] += 1
        else:
            nsp_data[label][str(nsp_label)] += 1


    return data, nsp_data


def get_data_for_bert_2(args, dna_sequences, labels):
    """ process data obtained from DNABERT """
    data = []
    dict_labels = defaultdict(int)
    for i in range(len(dna_sequences)):
        label = labels[i]
        # parse dna sequences
        dna_list = [args.dict_kmers['[CLS]']] + [args.dict_kmers[kmer] if kmer in args.dict_kmers else args.dict_kmers['[UNK]'] for kmer in dna_sequences[i]] + [args.dict_kmers['[SEP]']]
        segment_ids = [0] * args.max_read_length
        if args.bert_step == 'finetuning':
            # pad input ids vector if necessary
            if len(dna_list) < args.max_read_length:
                num_padded_values = args.max_read_length-len(dna_list)
                dna_list = dna_list + [args.dict_kmers['[PAD]']] * num_padded_values
                # create input_mask vector indicating padded values. Padding token indices are masked (0) to avoid
                # performing attention on them.
                input_mask = [1]*(args.max_read_length-num_padded_values) + [0]*num_padded_values
            else:
                input_mask = [1]*args.max_read_length
            data.append([dna_list, input_mask, segment_ids, label])
            dict_labels[label] += 1
            # print(r, dna_list, input_mask, segment_ids, label)


    return data, dict_labels


def create_tfrecords(args, data):
    # for fq_file in grouped_files:
    """ Converts dna sequences to tfrecord """
    num_lines = 8 if args.pair else 4
    output_prefix = '.'.join(args.input.split('/')[-1].split('.')[0:-1])
    output_tfrec = os.path.join(args.output_dir, output_prefix + '.tfrec')
    count = 0

    if args.dnabert:
        dna_sequences, labels = data
    else:
        # data = fastq file
        with open(data) as handle:
            content = handle.readlines()
            reads = [''.join(content[j:j + num_lines]) for j in range(0, len(content), num_lines)]
            dna_sequences = [r.rstrip().split('\n')[1] for r in reads]
            if args.update_labels:
                labels = [int(args.labels_mapping[r.rstrip().split('\n')[0].split('|')[1]]) for r in reads]
            else:
                labels = [int(r.rstrip().split('\n')[0].split('|')[1]) for r in reads]
            # random.shuffle(reads)
            print(f'{fq_file}\t# reads: {len(reads)}\t{len(dna_sequences)}')

    if args.bert:
        # # create processes
        # chunk_size = math.ceil(len(reads)/args.num_proc)
        # grouped_reads = [reads[i:i+chunk_size] for i in range(0, len(reads), chunk_size)]
        # indices = list(range(len(reads)))
        # grouped_reads_index = [indices[i:i+chunk_size] for i in range(0, len(indices), chunk_size)]

        # with mp.Manager() as manager:
        #     data = manager.dict()
        #     nsp_data = manager.dict()
        #     if args.dataset_type == 'sim':
        #         processes = [mp.Process(target=get_data_for_bert, args=(args, nsp_data, data, reads, grouped_reads[i], grouped_reads_index[i], i)) for i in range(len(grouped_reads))]
        #     for p in processes:
        #         p.start()
        #     for p in processes:
        #         p.join()
        
        if not args.dnabert:
            reads_index = list(range(len(reads)))
            data, nsp_data = get_data_for_bert_1(args, dna_sequences, labels, reads_index)

            total_reads = 0
            nsp_1_labels = defaultdict(int)
            nsp_0_labels = defaultdict(int)
            for label, nsp_count in nsp_data.items():
                nsp_1_labels[label]+= nsp_count['1']
                nsp_0_labels[label]+= nsp_count['0']
                total_reads += nsp_count['1'] + nsp_count['0']
                # for process, nsp_data_process in nsp_data.items():
                #     for label, nsp_count in nsp_data_process.items():
                #         nsp_1_labels[label]+= nsp_count['1']
                #         nsp_0_labels[label]+= nsp_count['0']
                #         total_reads += nsp_count['1'] + nsp_count['0']
            with open(os.path.join(args.output_dir, 'nsp_0_data_info.json'), 'w') as nsp_f:    
                json.dump(nsp_0_labels, nsp_f)
            with open(os.path.join(args.output_dir, 'nsp_1_data_info.json'), 'w') as nsp_f:    
                json.dump(nsp_1_labels, nsp_f)
            print(f'total reads: {total_reads}')

        elif args.dnabert:
            data, dict_labels = get_data_for_bert_2(args, dna_sequences, labels)
            print(f'dict_labels: {dict_labels}')

        with tf.io.TFRecordWriter(output_tfrec) as writer:
                # for process, data_process in data.items():
                #     print(process, len(data_process))
            for i, r in enumerate(data, 0):
                if i == 0 and args.bert_step == 'pretraining':
                    print(f'{len(r)}\tinput_ids: {r[0]}\tinput_mask: {r[1]}\tsegment_ids: {r[2]}\tmasked_lm_positions: {r[3]}\n')
                    print(f'masked_lm_weights: {r[4]}\tmasked_lm_ids: {r[5]}\tnext_sentence_labels: {r[6]}')
                    print(f'input_ids: {len(r[0])}\tinput_mask: {len(r[1])}\tsegment_ids: {len(r[2])}\tmasked_lm_positions: {len(r[3])}\n')
                    print(f'masked_lm_weights: {len(r[4])}\tmasked_lm_ids: {len(r[5])}')
                if i == 0 and args.bert_step == 'finetuning':
                    print(f'{len(r)}\tinput_ids: {r[0]}\tinput_mask: {r[1]}\tsegment_ids: {r[2]}\tlabels: {r[3]}\n')
                    print(f'{len(r)}\tinput_ids: {len(r[0])}\tinput_mask: {len(r[1])}\tsegment_ids: {len(r[2])}\n')
                """
                input_ids: vector with ids by tokens (includes masked tokens: MASK, original, random) - input_ids
                input_mask: [1]*len(input_ids) - input_mask
                segment_ids: vector indicating the first (0) from the second (1) part of the sequence - segment_ids
                masked_lm_positions: positions of masked tokens (0 for padded values) - masked_lm_positions
                masked_lm_ids: original ids of masked tokens (0 for padded values) - masked_lm_ids
                masked_lm_weights: [1.0]*len(masked_lm_ids) (0.0 for padded values) - masked_lm_weights
                next_sentence_labels: 0 for "is not next" and 1 for "is next" - nsp_label
                label: species label - label
                """
                if args.bert_step == 'pretraining':
                    tfrecord_data = \
                        {
                            'input_ids': wrap_read(r[0]),
                            'attention_mask': wrap_read(r[1]),
                            'token_type_ids': wrap_read(r[2]),
                            'masked_lm_positions': wrap_read(r[3]),
                            'masked_lm_weights': wrap_weights(r[4]),
                            'masked_lm_ids': wrap_read(r[5])
                            # 'next_sentence_label': wrap_label(r[6]),
                        }
                elif args.bert_step == 'finetuning':
                    tfrecord_data = \
                        {
                            'input_ids': wrap_read(r[0]),
                            'attention_mask': wrap_read(r[1]),
                            'token_type_ids': wrap_read(r[2]),
                            'labels': wrap_label(r[3])
                            # 'is_real_example': wrap_label(1)
                        }
                feature = tf.train.Features(feature=tfrecord_data)
                example = tf.train.Example(features=feature)
                serialized = example.SerializeToString()
                writer.write(serialized)
                count += 1
                
        with open(os.path.join(args.output_dir, output_prefix + '-read_count'), 'w') as f:
            f.write(f'{count}')
    
    else:
        with tf.io.TFRecordWriter(output_tfrec) as writer:
            for i, r in enumerate(dna_sequences, 0):
                label = labels[i]
                if args.dnabert:
                    dna_list = [args.dict_kmers[kmer] if kmer in args.dict_kmers else args.dict_kmers['[UNK]'] for kmer in dna_sequences[i]]
                else:
                    dna_list  = prepare_input_data(args, dna_sequences[i])              
                # create TFrecords
                if args.no_label:
                    tfrecord_data = \
                        {
                            'read': wrap_read(dna_list),
                        }
                else:
                    tfrecord_data = \
                        {
                            'read': wrap_read(dna_list),
                            'label': wrap_label(label),
                        }
                feature = tf.train.Features(feature=tfrecord_data)
                example = tf.train.Example(features=feature)
                serialized = example.SerializeToString()
                writer.write(serialized)
                count += 1

            with open(os.path.join(args.output_dir, output_prefix + '-read_count'), 'w') as f:
                f.write(f'{count}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help="Path to the input fastq file or directory containing fastq files")
    parser.add_argument('--output_dir', help="Path to the output directory", default=os.getcwd())
    parser.add_argument('--vocab', help="Path to the vocabulary file")
    parser.add_argument('--DNA_model', action='store_true', default=False, help="represent reads for DNA model")
    parser.add_argument('--bert', action='store_true', default=False, help="process reads for transformer")
    parser.add_argument('--bert_step', help="choose between pre-training or fine-tuning task", choices=['pretraining','finetuning'])
    parser.add_argument('--no_label', action='store_true', default=False, help="do not add labels to tfrecords")
    parser.add_argument('--insert_size', action='store_true', default=False, help="add insert size info")
    parser.add_argument('--pair', action='store_true', default=False, help="represent reads as pairs")
    parser.add_argument('--dnabert', action='store_true', default=False, help="process dnabert data")
    parser.add_argument('--k_value', default=1, type=int, help="Size of k-mers")
    parser.add_argument('--num_proc', default=1, type=int, help="number of processes")
    parser.add_argument('--masked_lm_prob', default=0.15, type=float, help="Fraction of masked tokens in mlm task")
    parser.add_argument('--step', default=1, type=int, help="Length of step when sliding window over read")
    parser.add_argument('--update_labels', action='store_true', default=False, required=('--mapping_file' in sys.argv))
    parser.add_argument('--contiguous', action='store_true', default=False)
    parser.add_argument('--mapping_file', type=str, help='path to file mapping species labels to rank labels')
    parser.add_argument('--read_length', default=250, type=int, help="The length of simulated reads")
    parser.add_argument('--max_read_length', default=510, type=int, help="maximum length of dna sequences", required=('--dnabert' in sys.argv))
    parser.add_argument('--dataset_type', type=str, help="type of dataset", choices=['sim', 'meta'])
    args = parser.parse_args()

    print(args)

    if args.update_labels:
        args.labels_mapping = dict()
        with open(args.mapping_file, 'r') as f:
            for line in f:
                args.labels_mapping[line.rstrip().split('\t')[0]] = line.rstrip().split('\t')[1]

    if not args.bert:
        if os.path.isdir(args.input):
            # get list of fastq files
            fq_files = glob.glob(os.path.join(args.input, "*.fq"))
        else:
            fq_files = [args.input]
            args.num_proc = 1

    if not args.DNA_model:
        if args.bert:
            # compute size of output data for input_word_ids (CLS and SEP tokens are added with SEP at the end)
            args.kmer_vector_length = args.read_length//2 - args.k_value + 1
            print(f'final input vector length (without NSP task): {args.kmer_vector_length*2 + 2}')
            # print(f'final input vector length: {args.kmer_vector_length*2 + 3}')
        elif args.dnabert:
            args.kmer_vector_length = args.max_read_length 
        else:
            args.kmer_vector_length = args.read_length - args.k_value + 1 if args.step == 1 else args.read_length // args.k_value
        # get dictionary mapping kmers to indexes
        args.dict_kmers = vocab_dict(args.vocab)
        # if args.bert:
        #     # add [PAD] to dictionary
        #     args.dict_kmers['[PAD]'] = 0
        #     print(args.dict_kmers)
        with open(os.path.join(args.output_dir, f'{args.k_value}-dict.json'), 'w') as f:
            json.dump(args.dict_kmers, f)

    if args.dnabert:
        dna_sequences, labels = load_dnabert_seq(args)
        create_tfrecords(args, (dna_sequences, labels))
    elif args.bert:
        create_tfrecords(args, args.input)
    else:
        chunk_size = math.ceil(len(fq_files)/args.num_proc)
        grouped_files = [fq_files[i:i+chunk_size] for i in range(0, len(fq_files), chunk_size)]
        with mp.Manager() as manager:
            train_reads = manager.dict()
            val_reads = manager.dict()
            if args.dataset_type == 'sim':
                processes = [mp.Process(target=create_tfrecords, args=(args, grouped_files[i])) for i in range(len(grouped_files))]
            elif args.dataset_type == 'meta':
                processes = [mp.Process(target=create_meta_tfrecords, args=(args, grouped_files[i])) for i in range(len(grouped_files))]
            for p in processes:
                p.start()
            for p in processes:
                p.join()


if __name__ == "__main__":
    main()
