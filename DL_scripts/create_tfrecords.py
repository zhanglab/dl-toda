import os
import numpy as np
from Bio import SeqIO
import tensorflow as tf
import argparse
import gzip
import sys
import math
from tfrecords_utils import vocab_dict, get_kmer_arr

def wrap_read(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def wrap_label(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_meta_tfrecords(args):
    """ Converts metagenomic reads to tfrecords """
    output_tfrec = os.path.join(args.output_dir, args.output_prefix + '.tfrec')
    outfile = open('/'.join([args.output_dir, args.output_prefix + f'-read_ids.tsv']), 'w')
    with tf.compat.v1.python_io.TFRecordWriter(output_tfrec) as writer:
        with gzip.open(args.input_fastq, 'rt') as handle:
            for count, rec in enumerate(SeqIO.parse(handle, 'fastq'), 1):
                read = str(rec.seq)
                read_id = rec.description
                outfile.write(f'{read_id}\t{count}\n')
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

            with open(os.path.join(args.output_dir, args.output_prefix + '-read_count'), 'w') as f:
                f.write(f'{count}')

    outfile.close()


def create_tfrecords(args):
    """ Converts simulated reads to tfrecord """
    output_tfrec = os.path.join(args.output_dir, args.output_prefix + '.tfrec')
    with tf.compat.v1.python_io.TFRecordWriter(output_tfrec) as writer:
        with open(args.input_fastq) as handle:
            for count, rec in enumerate(SeqIO.parse(handle, 'fastq'), 1):
                read = str(rec.seq)
                read_id = rec.id
                label = int(read_id.split('|')[1])
                kmer_array = get_kmer_arr(args, read)
                data = \
                    {
                        'read': wrap_read(kmer_array),
                        'label': wrap_label(label),
                    }
                feature = tf.train.Features(feature=data)
                example = tf.train.Example(features=feature)
                serialized = example.SerializeToString()
                writer.write(serialized)

        with open(os.path.join(args.output_dir, args.output_prefix + '-read_count'), 'w') as f:
            f.write(f'{count}')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fastq', help="Path to the input fastq file")
    parser.add_argument('--output_dir', help="Path to the output directory")
    parser.add_argument('--vocab', help="Path to the vocabulary file")
    parser.add_argument('--k_value', default=12, type=int, help="Size of k-mers")
    parser.add_argument('--read_length', default=250, type=int, help="The length of simulated reads")
    parser.add_argument('--dataset_type', type=str, help="Type of dataset", choices=['sim', 'meta'])

    args = parser.parse_args()
    args.output_prefix = '.'.join(args.input_fastq.split('/')[-1].split('.')[0:-1]) if args.dataset_type == 'sim' else '.'.join(args.input_fastq.split('/')[-1].split('.')[0:-2])
    args.kmer_vector_length = args.read_length - args.k_value + 1
    # get dictionary mapping kmers to indexes
    args.dict_kmers = vocab_dict(args.vocab)

    if args.dataset_type == 'sim':
        create_tfrecords(args)
    elif args.dataset_type == 'meta':
        create_meta_tfrecords(args)

if __name__ == "__main__":
    main()
