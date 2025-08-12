import os
import sys
import tensorflow as tf
import numpy as np
import random
import datetime
from collections import defaultdict
import statistics
import argparse
import json
import glob
import math
import gzip
import multiprocessing as mp
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from DL_scripts.tfrecords_utils import vocab_dict, get_kmer_arr, prepare_input_data
from DL_scripts.tfrecords_bert_utils import *


def wrap_vector(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def wrap_label(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_weights(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_meta_tfrecords(args, kmer_vector, writer, outfile):

    if args.bert:
        input_ids, attention_mask, position_ids, token_type_ids, input_ids_size = prepare_data_for_bert(args, kmer_vector)
        # print(input_ids, attention_mask, position_ids, token_type_ids, kmer_vector_size, input_ids_size)
        outfile.write(f'{len(kmer_vector)}\t{input_ids_size}\n')
        tfrecord_data = \
            {
                'input_ids': wrap_vector(input_ids),
                'attention_mask': wrap_vector(attention_mask),
                'position_ids': wrap_vector(position_ids),
                'token_type_ids': wrap_vector(token_type_ids),
            }
    else:
        if len(kmer_vector) < args.kmer_vector_length:
            num_padded_values = args.kmer_vector_length-len(kmer_vector)
            kmer_vector = kmer_vector + [args.dict_kmers['[PAD]']] * num_padded_values
        outfile.write(f'{len(kmer_vector)}\n')
        # print(kmer_vector, len(kmer_vector))
        tfrecord_data = \
            {
                'read': wrap_vector(kmer_vector),
            }

    feature = tf.train.Features(feature=tfrecord_data)
    example = tf.train.Example(features=feature)
    serialized = example.SerializeToString()
    writer.write(serialized)


def prepare_meta_data(args):
        output_prefix = '.'.join(args.input.split('/')[-1].split('.')[0:-2]) if args.input[-2:] == 'gz' else '.'.join(args.input.split('/')[-1].split('.')[0:-1])
        output_tfrec = os.path.join(args.output_dir, output_prefix + '.tfrec')
        outfile = open('/'.join([args.output_dir, output_prefix + f'-read_ids.tsv']), 'w')
        with tf.io.TFRecordWriter(output_tfrec) as writer:
            if args.input[-2:] == 'gz':
                handle = gzip.open(args.input, 'rt')
            else:
                handle = open(args.input, 'r')
            # with gzip.open(args.input_fastq, 'rt') as handle:
            content = handle.readlines()
            reads = [''.join(content[j:j+4]) for j in range(0, len(content), 4)]
            print(reads[:10])
            for count, rec in enumerate(reads, 1):
                read = rec.split('\n')[1].rstrip()
                read_id = rec.split('\n')[0].rstrip()
                dna_list = prepare_input_data(args, read)

                if len(dna_list) > args.kmer_vector_length:
                    num_parts = math.ceil(len(dna_list) / args.kmer_vector_length)
                    grouped_tokens = [dna_list[i:i+args.kmer_vector_length] for i in range(0, len(dna_list), args.kmer_vector_length)]

                    for kmer_vector in grouped_tokens:
                        # print(f'{read_id}\t{len(read)}\t{read}\t{len(dna_list)}\t{num_parts}\n')
                        outfile.write(f'{read_id}\t{len(read)}\t{num_parts}\t')
                        create_meta_tfrecords(args, kmer_vector, writer, outfile)

                else:
                    # print(f'{read_id}\t{len(read)}\t{read}\t{len(dna_list)}\t1\n')
                    outfile.write(f'{read_id}\t{len(read)}\t1\t')
                    create_meta_tfrecords(args, dna_list, writer, outfile)
                   
                # if count == 10:
                #     break

            with open(os.path.join(args.output_dir, output_prefix + '-read_count'), 'w') as f:
                f.write(f'{count}')

        outfile.close()


def prepare_data_for_bert(dna_list, kmer_vector_length, bert_step, masked_lm_prob, dict_kmers, contiguous_kmers=False):
    """ process data obtained from DNABERT """
    max_position_embeddings = 512 # define the maximum sequence length the model can encounter in the dataset

    # adjust size for sequences longer than the max read length (dnabert data generates sequences of size > 510 when specifying a size of 510!! je ne sais pas pourquoi)
    if len(dna_list) > kmer_vector_length: # --> max read length is 511 for dnabert data, just for k = 4 not k= 1
        dna_list = dna_list[:kmer_vector_length]
    
    if bert_step == 'pretraining':
        # compute the number of tokens to mask
        n_mlm = int(masked_lm_prob * len(dna_list))
        
        # get list of indices of tokens to mask
        mlm_positions = random.sample(list(range(len(dna_list))), n_mlm)

        if contiguous_kmers:
            # get indices of contiguous kmers (previous and following kmer)
            # indices_to_mask = [-1, 1, 2]
            indices_to_mask = [-1, 1]
            mlm_positions.sort()
            contiguous_positions = set()
            for mask_position in mlm_positions:
                for mask_index in indices_to_mask:
                    current_index = mask_position + mask_index
                    if current_index <= (len(dna_list)-1) and current_index >= 0:
                        # print(mask_position, mask_index, current_index)
                        contiguous_positions.add(current_index)
            
            # mask contiguous kmers
            mlm_positions += list(contiguous_positions)

        # n_masked_pos.append(len(mlm_positions)/len(dna_list))
        
        # mask tokens
        mlm_dna_list = get_masked_array(args, mlm_positions, dna_list)

        # define vector of labels containing indices of masked tokens and -100 for unmasked tokens
        mlm_labels = [dna_list[i] if i in mlm_positions else -100 for i in range(len(dna_list))]
        
        # define NSP label - NSP is not implemented here
        # next_sentence_label = 1
        input_ids = mlm_dna_list
        labels = mlm_labels

    else:
        input_ids = dna_list

    # add CLS and SEP tokens
    input_ids = [dict_kmers['[CLS]']] + input_ids + [dict_kmers['[SEP]']]

    if bert_step == 'pretraining':
        # update vector of labels to reflect the addition of the special tokens
        labels = [-100] + labels + [-100]

    # define the first and second part of the sequence - NSP is not implemented here
    token_type_ids = [0] * max_position_embeddings
    
    # pad input vectors if necessary
    if len(input_ids) < max_position_embeddings:
        num_padded_values = max_position_embeddings - len(input_ids)
        input_ids = input_ids + [dict_kmers['[PAD]']] * num_padded_values
        if bert_step == 'pretraining':
            labels = labels + [-100] * num_padded_values
        # create attention_mask vector indicating padded values. Padding token indices are masked (0) to avoid
        # performing attention on them.
        attention_mask = [1]*(max_position_embeddings - num_padded_values) + [0]*num_padded_values
    else:
        attention_mask = [1]*max_position_embeddings

    position_ids = list(range(max_position_embeddings))

    if bert_step == 'pretraining':
        return input_ids, attention_mask, position_ids, token_type_ids, labels, len(mlm_positions)/len(dna_list), len(dna_list), len(input_ids)
    else:
        return input_ids, attention_mask, position_ids, token_type_ids, len(input_ids)


def create_tfrecords(input_file, output_dir, k_value, step, read_length, kmer_vector_length, dict_kmers, labels_mapping, \
        masked_lm_prob, dnabert=False, update_labels=False, no_label=False, dataset_type='sim', bert_step=None, bert=False):
    # for fq_file in grouped_files:
    """ Converts dna sequences to tfrecord """
    # num_lines = 8 if args.pair else 4
    output_prefix = '.'.join(input_file.split('/')[-1].split('.')[0:-1])
    output_tfrec = os.path.join(output_dir, output_prefix + '.tfrec')
    count = 0
    vector_size = set()
    dna_sequence_size = set()

    if bert:
        # create tfrecords for bert model
        if bert_step == 'pretraining':
            # monitor the fraction of masked positions
            n_masked_pos = []
        
        with tf.io.TFRecordWriter(output_tfrec) as writer:
            with open(input_file, 'r') as f:
                for count, line in enumerate(f):
                    if dnabert:
                        label = line.rstrip().split('\t')[0]
                        # dna_sequence = line.rstrip().split('\t')[1].split(" ")
                        dna_sequence = line.rstrip().split('\t')[3]
                        dna_list = prepare_input_data(dna_sequence, k_value, step, read_length, dict_kmers, dataset_type=dataset_type) 
                        # parse dna sequence into kmers
                        # dna_list = [dict_kmers[kmer] if kmer in dict_kmers else dict_kmers['[UNK]'] for kmer in dna_sequence]
                    else:
                        label = line.rstrip().split('\t')[0].split('|')[1]
                        dna_sequence = line.rstrip().split('\t')[1]
                        # parse dna sequence into kmers
                        dna_list = prepare_input_data(dna_sequence, k_value, step, read_length, dict_kmers, dataset_type=dataset_type)  
                    if update_labels:
                        label = int(labels_mapping[label])
                    else:
                        label = int(label)

                    if count == 0:
                        reconstructed_token_list = []
                        for token_id in dna_list:
                            for key, value in dict_kmers.items():
                                if value == token_id:
                                    reconstructed_token_list.append(key)
                        with open(os.path.join(output_dir, output_prefix + '-example-sequence-1'), 'w') as out_ex:
                            out_ex.write(f'count\t{count+1}\nline:\t{line}\nupdated label\t{label}'
                                f'\ndna sequence\t{dna_sequence}\ndna list\t{dna_list}\nreconstructed list of tokens\t{reconstructed_token_list}')

                    """
                    input_ids: vector with indices of tokens (includes masked token: MASK) - length: 512
                    attention_mask: vector necessary to avoid performing attention on padded positions (0 for positions with the PAD token and 1 otherwise)  - length: 512
                    token_type_ids: vector indicating the first (0) from the second (1) part of the sequence - length: 512
                    # masked_lm_positions: positions of masked tokens (0 for padded values) - masked_lm_positions
                    # masked_lm_ids: original ids of masked tokens (0 for padded values) - masked_lm_ids
                    # masked_lm_weights: [1.0]*len(masked_lm_ids) (0.0 for padded values) - masked_lm_weights
                    next_sentence_label: 0 for "is not next" and 1 for "is next" - nsp_label
                    labels: labels for computing the MLM loss (indices of tokens for masked tokens and -100 for unmasked tokens)  - length: 512
                    """
                    if bert_step == 'pretraining':
                        input_ids, attention_mask, position_ids, token_type_ids, labels, fraction_masked_pos, sequence_size = prepare_data_for_bert(dna_list, kmer_vector_length, bert_step, masked_lm_prob, dict_kmers)
                        n_masked_pos.append(fraction_masked_pos)
                        tfrecord_data = \
                            {
                                'input_ids': wrap_vector(input_ids),
                                'attention_mask': wrap_vector(attention_mask),
                                'position_ids': wrap_vector(position_ids),
                                'token_type_ids': wrap_vector(token_type_ids),
                                'labels': wrap_vector(labels),
                                # 'next_sentence_label': wrap_label(r[4])
                            }
                    elif bert_step in ['finetuning','regular']:
                        input_ids, attention_mask, position_ids, token_type_ids, sequence_size = prepare_data_for_bert(dna_list, kmer_vector_length, bert_step, masked_lm_prob, dict_kmers)
                        tfrecord_data = \
                            {
                                'input_ids': wrap_vector(input_ids),
                                'attention_mask': wrap_vector(attention_mask),
                                'position_ids': wrap_vector(position_ids),
                                'token_type_ids': wrap_vector(token_type_ids),
                                'labels': wrap_label(label)
                            }
                    feature = tf.train.Features(feature=tfrecord_data)
                    example = tf.train.Example(features=feature)
                    serialized = example.SerializeToString()
                    writer.write(serialized)
                    count += 1  
                    vector_size.add(sequence_size)
                    dna_sequence_size.add(len(dna_sequence))

        if bert_step == 'pretraining':
            with open(info, 'w') as f:
                f.write(f'{min(n_masked_pos)}\t{max(n_masked_pos)}\t{statistics.mean(n_masked_pos)}\t{statistics.median(n_masked_pos)}')          
        
    else:
        # create tfrecords for cnn model
        with tf.io.TFRecordWriter(output_tfrec) as writer:
            with open(input_file, 'r') as f:
                for line in f:
                    if dnabert:
                        label = line.rstrip().split('\t')[0]
                        dna_sequence = line.rstrip().split('\t')[3]
                        dna_list = prepare_input_data(dna_sequence, k_value, step, read_length, dict_kmers) 
                        # dna_sequence = line.rstrip().split('\t')[1].split(" ")
                        # parse dna sequence into kmers
                        # dna_list = [args.dict_kmers[kmer] if kmer in dict_kmers else dict_kmers['[UNK]'] for kmer in dna_sequence]
                        
                        if len(dna_list) < kmer_vector_length:
                            num_padded_values = kmer_vector_length-len(dna_list)
                            dna_list = dna_list + [dict_kmers['[PAD]']] * num_padded_values
                        if len(dna_list) > kmer_vector_length: # --> max read length is 511 for dnabert data, just for k = 4 not k= 1
                            dna_list = dna_list[:kmer_vector_length] # remove the last kmer == information about the last nucleotide
    
                    else:
                        label = line.rstrip().split('\t')[0].split('|')[1]
                        dna_sequence = line.rstrip().split('\t')[1]
                        # parse dna sequence into kmers
                        dna_list = prepare_input_data(dna_sequence, k_value, step, read_length, dict_kmers)
                        if len(dna_list) < kmer_vector_length:
                            num_padded_values = kmer_vector_length-len(dna_list)
                            dna_list = dna_list + [dict_kmers['[PAD]']] * num_padded_values

                    if args.update_labels:
                        label = int(labels_mapping[label])

                    if count == 0:
                        reconstructed_token_list = []
                        for token_id in dna_list:
                            for key, value in dict_kmers.items():
                                if value == token_id:
                                    reconstructed_token_list.append(key)
                        with open(os.path.join(output_dir, output_prefix + '-example-sequence-1'), 'w') as out_ex:
                            out_ex.write(f'count\t{count+1}\nline:\t{line}\nupdated label\t{label}'
                                f'\ndna sequence\t{dna_sequence}\ndna list\t{dna_list}\nreconstructed list of tokens\t{reconstructed_token_list}')

                    # create TFrecords
                    if no_label:
                        tfrecord_data = \
                            {
                                'read': wrap_vector(dna_list),
                            }
                    else:
                        tfrecord_data = \
                            {
                                'read': wrap_vector(dna_list),
                                'label': wrap_label(label),
                            }
                    feature = tf.train.Features(feature=tfrecord_data)
                    example = tf.train.Example(features=feature)
                    serialized = example.SerializeToString()
                    writer.write(serialized)
                    count += 1
                    vector_size.add(len(dna_list))
                    dna_sequence_size.add(len(dna_sequence))

        # else:
        #     with tf.io.TFRecordWriter(output_tfrec) as writer:
        #         with open(args.input, 'r') as f:
        #             line_count = 1

        #             for line in f:
        #                 if line_count == 1:
        #                     label = line.rstrip().split('|')[1]
        #                 elif line_count == 2:
        #                     dna_sequence = line.rstrip()
        #                 elif line_count == 4:         
        #                     # parse dna sequence into kmers
        #                     dna_list = prepare_input_data(args, dna_sequence)
        #                     if args.update_labels:
        #                         label = int(args.labels_mapping[label])
        #                     # create TFrecords
        #                     if args.no_label:
        #                         tfrecord_data = \
        #                             {
        #                                 'read': wrap_vector(dna_list),
        #                             }
        #                     else:
        #                         tfrecord_data = \
        #                             {
        #                                 'read': wrap_vector(dna_list),
        #                                 'label': wrap_label(label),
        #                             }
        #                     feature = tf.train.Features(feature=tfrecord_data)
        #                     example = tf.train.Example(features=feature)
        #                     serialized = example.SerializeToString()
        #                     writer.write(serialized)
        #                     count += 1
        #                     vector_size.add(len(dna_list))
        #                     dna_sequence_size.add(len(dna_sequence))
        #                     line_count = 0 
                        
        #                 line_count += 1

    with open(os.path.join(output_dir, output_prefix + '-read_count'), 'w') as f:
        f.write(f'{count}')

    with open(os.path.join(output_dir, output_prefix + '-vector_size'), 'w') as f:
        f.write(f'min vector size: {min(list(vector_size))}\n')
        f.write(f'max vector size: {max(list(vector_size))}\n')
        f.write(f'mean vector size: {statistics.mean(list(vector_size))}\n')
        f.write(f'median vector size: {statistics.median(list(vector_size))}\n')
        f.write(f'max read length: {read_length}\n')


    with open(os.path.join(output_dir, output_prefix + '-dna_seq_size'), 'w') as f:
        f.write(f'min dna sequence size: {min(list(dna_sequence_size))}\n')
        f.write(f'max dna sequence size: {max(list(dna_sequence_size))}\n')
        f.write(f'mean dna sequence size: {statistics.mean(list(dna_sequence_size))}\n')
        f.write(f'median dna sequence size: {statistics.median(list(dna_sequence_size))}\n')
        f.write(f'max read length: {read_length}\n')

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
    parser.add_argument('--canonical_kmers', action='store_true', default=False, help="use a vocabulary made of canonical kmers")
    parser.add_argument('--k_value', type=int, help="Size of k-mers", required=True)
    parser.add_argument('--num_proc', default=1, type=int, help="number of processes")
    parser.add_argument('--masked_lm_prob', default=0.15, type=float, help="Fraction of masked tokens in mlm task")
    parser.add_argument('--step', default=1, type=int, help="Length of step when sliding window over read")
    parser.add_argument('--update_labels', action='store_true', default=False, required=('--mapping_file' in sys.argv))
    parser.add_argument('--contiguous_kmers', action='store_true', default=False)
    parser.add_argument('--mapping_file', type=str, help='path to file mapping species labels to rank labels')
    parser.add_argument('--max_read_length', default=250, type=int, help="The length of simulated reads", required=True)
    parser.add_argument('--dataset_type', type=str, help="type of dataset", choices=['sim', 'meta'])
    args = parser.parse_args()

    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.update_labels:
        labels_mapping = dict()
        with open(args.mapping_file, 'r') as f:
            for line in f:
                labels_mapping[line.rstrip().split('\t')[0]] = line.rstrip().split('\t')[1]

    if not args.DNA_model:
        args.kmer_vector_length = args.max_read_length - args.k_value + 1 if args.step == 1 else args.max_read_length // args.k_value
        print(f'max read length: {args.max_read_length}\tvector size: {args.kmer_vector_length}\t{args.k_value}')
        # get dictionary mapping kmers to indexes
        args.dict_kmers = vocab_dict(f'{args.vocab}/{args.k_value}mers.txt')
        with open(os.path.join(args.output_dir, f'{args.k_value}-dict.json'), 'w') as f:
            json.dump(args.dict_kmers, f)

    if args.dataset_type == "sim":
        create_tfrecords(input_file, args.output_dir, args.k_value, args.step, args.max_read_length, args.kmer_vector_length, args.dict_kmers, args.labels_mapping, \
        args.masked_lm_prob, dnabert=args.dnabert, update_labels=args.update_labels, bert_step=args.bert_step, no_label=args.no_label, dataset_type='sim')

    elif args.dataset_type == "meta":
        prepare_meta_data(args)

if __name__ == "__main__":
    main()
