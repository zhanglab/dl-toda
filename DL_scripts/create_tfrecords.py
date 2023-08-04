import os
import sys
import tensorflow as tf
import numpy as np
import random
# from Bio import SeqIO
import argparse
import gzip
from tfrecords_utils import vocab_dict, get_kmer_arr


def get_masked_kmers(args, kmer_array):
    # number of k-mers to mask in the vector of k-mers
    n_mask = int(0.15 * args.kmer_vector_length)
    # choose index of first k-mer to mask
    start_mask_idx = random.choice(range(0, args.kmer_vector_length - n_mask))
    print(start_mask_idx)
    # select k-mers to mask
    kmers_masked = [False if i not in range(start_mask_idx,start_mask_idx+n_mask) else True for i in range(args.kmer_vector_length)]
    # change labels for masked k-mers
    kmer_array_masked = np.copy(kmer_array)
    kmer_array_masked[kmers_masked] = args.dict_kmers['mask']
    print(kmer_array_masked)
    # prepare sample_weights parameter to loss function
    sample_weights = np.ones(kmer_array.shape)
    sample_weights[kmers_masked == False] = 0
    print(sample_weights)
    return kmer_array_masked, sample_weights


def wrap_read(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def wrap_label(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_meta_tfrecords(args):
    """ Converts metagenomic reads to tfrecords """
    output_prefix = '.'.join(args.input_fastq.split('/')[-1].split('.')[0:-2]) if args.input_fastq[-2:] == 'gz' else '.'.join(args.input_fastq.split('/')[-1].split('.')[0:-1])
    output_tfrec = os.path.join(args.output_dir, output_prefix + '.tfrec')
    outfile = open('/'.join([args.output_dir, output_prefix + f'-read_ids.tsv']), 'w')
    with tf.compat.v1.python_io.TFRecordWriter(output_tfrec) as writer:
        if args.input_fastq[-2:] == 'gz':
            handle = gzip.open(args.fastq, 'rt')
        else:
            handle = open(args.input_fastq, 'r')
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


def create_tfrecords(args):
    """ Converts simulated reads to tfrecord """
    output_prefix = '.'.join(args.input_fastq.split('/')[-1].split('.')[0:-1])
    output_tfrec = os.path.join(args.output_dir, output_prefix + '.tfrec')
    bases = {'A': 2, 'T': 3, 'C': 4, 'G': 5}
    num_lines = 8 if args.pair else 4
    count = 0
    with tf.io.TFRecordWriter(output_tfrec) as writer:
    # with tf.compat.v1.python_io.TFRecordWriter(output_tfrec) as writer:
        with open(args.input_fastq) as handle:
            rec = ''
            n_line = 0
            for line in handle:
                rec += line
                n_line += 1
                if n_line == num_lines:
            # content = handle.readlines()
            # reads = [''.join(content[j:j + num_lines]) for j in range(0, len(content), num_lines)]
            # print(f'{args.input_fastq}\t# reads: {len(reads)}')
            # for count, rec in enumerate(reads, 1):
            # for count, rec in enumerate(SeqIO.parse(handle, 'fastq'), 1):
            #     read = str(rec.seq)
            #     read_id = rec.id
                # label = int(read_id.split('|')[1])
                    read_id = rec.split('\n')[0].rstrip()
                    label = int(read_id.split('|')[1])
                    print(read_id)
                    # update label if necessary
                    if args.update_labels:
                        label = int(args.labels_mapping[str(label)])
                    if args.pair:
                        if args.DNA_model:
                            fw_dna_array = [bases[x] if x in bases else 1 for x in rec.split('\n')[1].rstrip()]
                            rv_dna_array = [bases[x] if x in bases else 1 for x in rec.split('\n')[5].rstrip()]
                            # update read length to match the max read length
                            if len(fw_dna_array) < args.read_length:
                                # pad list of bases with 0s to the right
                                fw_dna_array = fw_dna_array + [0] * (args.read_length - len(fw_dna_array))
                                rv_dna_array = rv_dna_array + [0] * (args.read_length - len(rv_dna_array))
                        else:
                            fw_dna_array = get_kmer_arr(args, rec.split('\n')[1].rstrip())
                            rv_dna_array = get_kmer_arr(args, rec.split('\n')[5].rstrip())

                        dna_array = fw_dna_array + rv_dna_array
                        # append insert size for kmers arrays as pairs of reads
                        if args.insert_size:
                            dna_array.append(int(args.dict_kmers[read_id.split('|')[3]]))
                    else:
                        if args.DNA_model:
                            dna_array = [bases[x] if x in bases else 1 for x in rec.split('\n')[1].rstrip()]
                            # update read length to match the max read length
                            if len(dna_array) < args.read_length:
                                # pad list of bases with 0s to the right
                                dna_array = dna_array + [0] * (args.read_length - len(dna_array))
                        else:
                            dna_array = get_kmer_arr(args, rec.split('\n')[1].rstrip())

                    if args.no_label:
                        data = \
                            {
                                # 'read': wrap_read(np.array(dna_array)),
                                'read': wrap_read(dna_array),
                            }
                    if args.transformer:
                        # mask 15% of k-mers in reads
                        kmer_array_masked, sample_weights = get_masked_kmers(args, np.array(dna_array))
                        data = \
                            {
                                # 'read': wrap_read(np.array(dna_array)),
                                'read': wrap_read(dna_array),
                                'read_masked': wrap_read(kmer_array_masked),
                                'sample_weights': wrap_read(sample_weights),
                            }
                    else:
                        # record_bytes = tf.train.Example(features=tf.train.Features(feature={
                        #     "read": tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(dna_array))),
                        #     "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                        # })).SerializeToString()
                        # writer.write(record_bytes)

                        data = \
                            {
                                # 'read': wrap_read(np.array(dna_array)),
                                'read': wrap_read(dna_array),
                                'label': wrap_label(label),
                            }
                    feature = tf.train.Features(feature=data)
                    example = tf.train.Example(features=feature)
                    serialized = example.SerializeToString()
                    writer.write(serialized)
                    count += 1
                    # initialize variables again
                    n_line = 0
                    rec = ''

        with open(os.path.join(args.output_dir, output_prefix + '-read_count'), 'w') as f:
            f.write(f'{count}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fastq', help="Path to the input fastq file")
    parser.add_argument('--output_dir', help="Path to the output directory")
    parser.add_argument('--vocab', help="Path to the vocabulary file")
    parser.add_argument('--DNA_model', action='store_true', default=False, help="represent reads for DNA model")
    parser.add_argument('--transformer', action='store_true', default=False, help="represent reads for transformer")
    parser.add_argument('--no_label', action='store_true', default=False, help="do not add labels to tfrecords")
    parser.add_argument('--insert_size', action='store_true', default=False, help="add insert size info")
    parser.add_argument('--pair', action='store_true', default=False, help="represent reads as pairs")
    parser.add_argument('--k_value', default=12, type=int, help="Size of k-mers")
    parser.add_argument('--update_labels', action='store_true', default=False, required=('--mapping_file' in sys.argv))
    parser.add_argument('--mapping_file', type=str, help='path to file mapping species labels to rank labels')
    parser.add_argument('--read_length', default=250, type=int, help="The length of simulated reads")
    parser.add_argument('--dataset_type', type=str, help="Type of dataset", choices=['sim', 'meta'])

    args = parser.parse_args()
    if not args.DNA_model:
        args.kmer_vector_length = args.read_length - args.k_value + 1
        # get dictionary mapping kmers to indexes
        args.dict_kmers = vocab_dict(args.vocab)
    print(args.dict_kmers)

    if args.update_labels:
        args.labels_mapping = dict()
        with open(args.mapping_file, 'r') as f:
            for line in f:
                args.labels_mapping[line.rstrip().split('\t')[0]] = line.rstrip().split('\t')[1]

    if args.dataset_type == 'sim':
        create_tfrecords(args)
    elif args.dataset_type == 'meta':
        create_meta_tfrecords(args)


if __name__ == "__main__":
    main()
