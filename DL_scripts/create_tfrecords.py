import os
import sys
import tensorflow as tf
import numpy as np
import random
# from Bio import SeqIO
import argparse
import json
import glob
import math
import gzip
import multiprocessing as mp
from tfrecords_utils import vocab_dict, get_kmer_arr


def get_nsp_input(args, bases_list, pad_list):
    # create 2 segments from list
    segment_1 = bases_list[:len(bases_list)//2]
    segment_2 = bases_list[len(bases_list)//2:]
    # randomly choose whether to have segment 2 after segment 1 or not 
    nsp_choice = random.choice([True, False])
    # generate random segment 2 in case nsp_choice is False
    if nsp_choice:
        nsp_label = 0 # 'IsNext'
    else:
        nsp_label = 1 # 'NotNext'
        segment_2 = [args.dict_kmers[x] for x in ''.join(random.choices(['A','T','C','G'], k=len(segment_2)))]
    # concatenate segments
    concatenate_segments = [args.dict_kmers['CLS']] + segment_1 + [args.dict_kmers['SEP']] + segment_2 + [args.dict_kmers['SEP']]
    # update list of pad/non-pad values
    up_pad_list = [1] + pad_list[0:len(segment_1)] + [1] + pad_list[len(segment_1):len(pad_list)] +[1]
    # create list of segment ids
    segment_ids = [0]*(2+len(segment_1)) + [1]*(1+len(segment_2))
    
    return concatenate_segments, segment_ids, nsp_label, up_pad_list

def get_masked_array(args, mask_indexes, input_array):
    # replace each chosen base by the MASK token number 80% of the time, a random base 10% of the time
    # or the unchanged base 10% of the time
    replacements = ["masked", "random", "same"]
    weights = [0.8, 0.1, 0.1]
    for i in range(len(input_array)):
        if i in mask_indexes:
            # randomly choose one type of replacement
            r_type = random.choices(replacements, weights=weights)
            if r_type == 'masked':
                input_array[i] = args.dict_kmers["MASK"]
            elif r_type == 'random':
                input_array[i] = random.choices([args.dict_kmers[k] for k in args.dict_kmers.keys() if k not in ["UNK", "MASK", "CLS", "SEP"]])
            else:
                continue

    return input_array

def get_mlm_input(args, input_array):
    # compute number of bases to mask (take into account 2*'SEP' and 'CLS')
    n_mask = int(0.15 * (len(input_array)-3)) # --> could be updated to mask predictions per sequence
    
    # if args.contiguous:
    #     # mask contiguous bases
    #     # choose index of first base to mask
    #     start_mask_idx = random.choice(range(0, args.kmer_vector_length - n_mask))
    #     mask_indexes = list(range(start_mask_idx,start_mask_idx+n_mask))
    #     # select bases to mask
    #     bases_masked = [False if i not in range(start_mask_idx,start_mask_idx+n_mask) else True for i in range(args.kmer_vector_length)]
    # else:
    # get indexes of SEP, CLS and UNK tokens
    sep_indices = [i for i in range(len(input_array)) if input_array[i] in [args.dict_kmers['SEP'],args.dict_kmers['CLS'],args.dict_kmers['UNK']]]
    print(f'sep_indices: {sep_indices}')
    # get list of indices of tokens to mask
    mask_indexes = random.sample(list(set(range(len(input_array))) - set(sep_indices)), n_mask)
    print(f'mask_indexes: {mask_indexes}')
    # select bases to mask
    bases_masked = [False if i not in mask_indexes else True for i in range(len(input_array))]
    print(f'bases_masked: {bases_masked}')
    # mask bases
    masked_bases_array = get_masked_array(args, mask_indexes, np.copy(input_array))
    print(f'masked_bases_array: {masked_bases_array}')
    # prepare sample_weights parameter to loss function
    weights = np.ones(n_mask)
    # sample_weights = np.zeros(input_array.shape) 
    # sample_weights[bases_masked] = 1 # only compute loss for masked k-mers

    return masked_bases_array, weights, mask_indexes, [input_array[i] for i in mask_indexes]
    

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


def create_tfrecords(args, grouped_files):
    for fq_file in grouped_files:
        """ Converts simulated reads to tfrecord """
        output_prefix = '.'.join(fq_file.split('/')[-1].split('.')[0:-1])
        output_tfrec = os.path.join(args.output_dir, output_prefix + '.tfrec')
        bases = {'A': 2, 'T': 3, 'C': 4, 'G': 5}
        num_lines = 8 if args.pair else 4
        count = 0
        with tf.io.TFRecordWriter(output_tfrec) as writer:
        # with tf.compat.v1.python_io.TFRecordWriter(output_tfrec) as writer:
            with open(fq_file) as handle:
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
                        # update label if necessary
                        if args.update_labels:
                            label = int(args.labels_mapping[str(label)])
                        if args.pair:
                            # concatenate forward and reverse reads into one vector
                            if args.DNA_model:
                                fw_dna_array = [bases[x] if x in bases else 1 for x in rec.split('\n')[1].rstrip()]
                                rv_dna_array = [bases[x] if x in bases else 1 for x in rec.split('\n')[5].rstrip()]
                                # update read length to match the max read length
                                if len(fw_dna_array) < args.read_length:
                                    # pad list of bases with 0s to the right
                                    fw_dna_array = fw_dna_array + [0] * (args.read_length - len(fw_dna_array))
                                    rv_dna_array = rv_dna_array + [0] * (args.read_length - len(rv_dna_array))
                            else:
                                fw_dna_array, _ = get_kmer_arr(args, rec.split('\n')[1].rstrip())
                                rv_dna_array, _ = get_kmer_arr(args, rec.split('\n')[5].rstrip())

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
                                dna_array, pad_array = get_kmer_arr(args, rec.split('\n')[1].rstrip())
                        
                        # create TFrecords
                        if args.no_label:
                            data = \
                                {
                                    # 'read': wrap_read(np.array(dna_array)),
                                    'read': wrap_read(dna_array),
                                }
                        if args.bert:
                            # prepare input for next sentence prediction task
                            nsp_dna_array, segment_ids, nsp_label, up_pad_list = get_nsp_input(args, dna_array, pad_array)
                            print(nsp_dna_array, segment_ids, nsp_label, up_pad_list)
                            # mask 15% of k-mers in reads
                            masked_array, masked_weights, masked_positions, masked_ids = get_mlm_input(args, np.array(nsp_dna_array))
                            print(f'nsp_dna_array: {nsp_dna_array}\nmasked_array: {masked_array}\nmasked_weights: {masked_weights}\nmasked_positions: {masked_positions}\nmasked_ids: {masked_ids}')
                            """
                            nsp_dna_array: vector of bases
                            masked_array:
                            masked_weights: 
                            masked_positions: 
                            masked_ids:
                            segment_ids:
                            label: 
                            """
                            data = \
                                {
                                    'base_id_data': wrap_read(masked_array),
                                    'type_id_data': wrap_read(segment_ids),
                                    'pad_data': wrap_read(up_pad_list),
                                    'masked_weights': wrap_weights(masked_weights),
                                    'masked_positions': wrap_read(masked_positions),
                                    'masked_ids': wrap_read(masked_ids),
                                    'nsp_label': wrap_label(nsp_label),
                                    'label': wrap_label(label)
                                }
                            if count == 100:
                                break
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
                        # break
                        


            with open(os.path.join(args.output_dir, output_prefix + '-read_count'), 'w') as f:
                f.write(f'{count}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help="Path to the input fastq file or directory containing fastq files")
    parser.add_argument('--output_dir', help="Path to the output directory", default=os.getcwd())
    parser.add_argument('--vocab', help="Path to the vocabulary file")
    parser.add_argument('--DNA_model', action='store_true', default=False, help="represent reads for DNA model")
    parser.add_argument('--bert', action='store_true', default=False, help="represent reads for transformer")
    parser.add_argument('--no_label', action='store_true', default=False, help="do not add labels to tfrecords")
    parser.add_argument('--insert_size', action='store_true', default=False, help="add insert size info")
    parser.add_argument('--pair', action='store_true', default=False, help="represent reads as pairs")
    parser.add_argument('--k_value', default=1, type=int, help="Size of k-mers")
    parser.add_argument('--step', default=1, type=int, help="Length of step when sliding window over read")
    parser.add_argument('--update_labels', action='store_true', default=False, required=('--mapping_file' in sys.argv))
    parser.add_argument('--contiguous', action='store_true', default=False)
    parser.add_argument('--mapping_file', type=str, help='path to file mapping species labels to rank labels')
    parser.add_argument('--read_length', default=250, type=int, help="The length of simulated reads")
    parser.add_argument('--dataset_type', type=str, help="type of dataset", choices=['sim', 'meta'])
    args = parser.parse_args()

    if args.update_labels:
        args.labels_mapping = dict()
        with open(args.mapping_file, 'r') as f:
            for line in f:
                args.labels_mapping[line.rstrip().split('\t')[0]] = line.rstrip().split('\t')[1]

    if os.path.isdir(args.input):
        # get list of fastq files
        fq_files = glob.glob(os.path.join(args.input, "*.fq"))
        num_proc = mp.cpu_count()
    else:
        fq_files = [args.input]
        num_proc = 1

    if not args.DNA_model:
        args.kmer_vector_length = args.read_length - args.k_value + 1 if args.step == 1 else args.read_length // args.k_value
        # get dictionary mapping kmers to indexes
        args.dict_kmers = vocab_dict(args.vocab)
        with open(os.path.join(args.output_dir, f'{args.k_value}-dict.json'), 'w') as f:
            json.dump(args.dict_kmers, f)
        print(args.kmer_vector_length)

    chunk_size = math.ceil(len(fq_files)/num_proc)
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
