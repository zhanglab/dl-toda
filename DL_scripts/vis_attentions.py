import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertConfig
import os
import sys
import json
import glob
import numpy as np
import math
import argparse
import seaborn as sn
import pandas as pd
import statistics
import matplotlib.pyplot as plt 
from collections import defaultdict
import random
from pygenomeviz import GenomeViz


# set seed
seed = 42
# set seed for tensorflow
tf.random.set_seed(seed)
# set seed for numpy operations
np.random.seed(seed)
# set the global python random seed
random.seed(seed)


dl_toda_dir = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])

# disable eager execution
# tf.compat.v1.disable_eager_execution()
print(f'Is eager execution enabled: {tf.executing_eagerly()}')

# print which unit (CPU/GPU) is used for an operation
#tf.debugging.set_log_device_placement(True)

# enable XLA = XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that can accelerate
# TensorFlow models with potentially no source code changes
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


def build_dataset(args, filenames, num_classes, is_training, drop_remainder):

    def load_tfrecords_for_finetuning(proto_example):
        name_to_features = {
          "input_ids": tf.io.FixedLenFeature([args.vector_size], tf.int64),
          "attention_mask": tf.io.FixedLenFeature([args.vector_size], tf.int64),
          "token_type_ids": tf.io.FixedLenFeature([args.vector_size], tf.int64),
          "labels": tf.io.FixedLenFeature([], tf.int64)
        }
        parsed_example = tf.io.parse_single_example(serialized=proto_example, features=name_to_features)

        return {"input_ids": parsed_example['input_ids'], "token_type_ids": parsed_example['token_type_ids'], "attention_mask": parsed_example['attention_mask'], "labels": parsed_example['labels']}
        # return {"input_ids": parsed_example['input_ids'], "attention_mask": parsed_example['attention_mask'], "labels": parsed_example['labels']}

    def load_tfrecords_for_pretraining(proto_example):
        name_to_features = {
          "input_word_ids": tf.io.FixedLenFeature([args.vector_size], tf.int64),
          "input_mask": tf.io.FixedLenFeature([args.vector_size], tf.int64),
          "input_type_ids": tf.io.FixedLenFeature([args.vector_size], tf.int64),
          "masked_lm_positions": tf.io.FixedLenFeature([args.num_masked], tf.int64),
          "masked_lm_weights": tf.io.FixedLenFeature([args.num_masked], tf.float32),
          "masked_lm_ids": tf.io.FixedLenFeature([args.num_masked], tf.int64)
        }
        # load one example
        parsed_example = tf.io.parse_single_example(serialized=proto_example, features=name_to_features)
        input_word_ids = parsed_example['input_word_ids']
        input_mask = parsed_example['input_mask']
        input_type_ids = parsed_example['input_type_ids']
        masked_lm_positions = parsed_example['masked_lm_positions']
        masked_lm_weights = parsed_example['masked_lm_weights']
        masked_lm_ids = parsed_example['masked_lm_ids']

        return  (input_word_ids, input_mask, input_type_ids, masked_lm_positions, masked_lm_weights, masked_lm_ids)

    """ Return data in TFRecords """
    fn_load_data = {'finetuning': load_tfrecords_for_finetuning, 'pretraining': load_tfrecords_for_pretraining}

    dataset = tf.data.TFRecordDataset([filenames])

    if is_training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.map(map_func=fn_load_data[args.datatype])
    dataset = dataset.batch(args.batch_size, drop_remainder=drop_remainder)

    return dataset

@tf.function
def get_attentions(data, model, test_accuracy):
    outputs = model(**data, output_attentions=True)
    logits = model(**data).logits
    probs = tf.nn.softmax(logits, axis=-1)
    labels = data["labels"]
    test_accuracy.update_state(labels, probs)

    # get predicted labels and confidence scores
    pred_labels = tf.math.argmax(probs, axis=1)
    if tf.shape(probs)[1] == 2:
        pred_probs = probs
    else:
        pred_probs = tf.reduce_max(probs, axis=1)

    return outputs, pred_labels, pred_probs


def Normalize(x, x_min=0.0, x_max=np.inf):
    scaled_value = (x - x_min) / (x_max - x_min)
    return round(scaled_value, 3)*100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords', type=str, help='path to tfrecords', required=True)
    parser.add_argument('--tsv_file', type=str, help='path to tsv file', required=True)
    parser.add_argument('--output_dir', type=str, help='directory to store results', default=os.getcwd())
    parser.add_argument('--init_lr', type=float, help='initial learning rate', default=0.0001)
    parser.add_argument('--cutoff', type=float, help='cutoff for displaying attention scores', default=0.0)
    parser.add_argument('--batch_size', type=int, help='batch size per gpu', default=8192)
    parser.add_argument('--num_labels', type=int, help='number of labels', default=2)
    parser.add_argument('--k_value', type=int, help='length of kmer strings', default=12)
    parser.add_argument('--fn_read', type=str, help='false negative read id', required=True)
    parser.add_argument('--tp_read', type=str, help='true positive read id', required=True)
    parser.add_argument('--vocab', help="Path to the vocabulary file", required=('AlexNet' in sys.argv))
    parser.add_argument('--bert_config_file', type=str, help='path to bert config file', required=('BERT' in sys.argv or 'BERT_HUGGINGFACE' in sys.argv))
    parser.add_argument('--class_mapping', type=str, help='path to json file containing dictionary mapping taxa to labels', default=os.path.join(dl_toda_dir, 'data', 'species_labels.json'))
    parser.add_argument('--model', type=str, help='path to directory containing keras model saved with .save()')
    parser.add_argument('--pretrained', type=str, help='path to model saved with .save_pretrained()')
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus, 'GPU')

    # get vocabulary
    with open(f'{args.vocab}/{args.k_value}mers.txt', 'r') as f:
        content = f.readlines()
        vocab = {i: content[i].strip() for i in range(len(content))}
    print(vocab)

    # load class_mapping file mapping label IDs to species
    if args.class_mapping:
        f = open(args.class_mapping)
        class_mapping = json.load(f)
        num_labels = len(class_mapping)
    else:
        num_labels = args.num_labels

    # create output directories
    if not os.path.isdir(args.output_dir):
        os.makedirs(os.path.join(args.output_dir))


    init_lr = args.init_lr
    opt = tf.keras.optimizers.Adam(init_lr)


    # load model
    with open(args.bert_config_file, "r") as f:
        args.config_dict = json.load(f)

    bert_config = BertConfig(vocab_size=args.config_dict["vocab_size"])

    if args.model is not None:
        model = tf.keras.models.load_model(args.model)
    else:
        model = TFBertForSequenceClassification.from_pretrained(args.pretrained, config=bert_config)
    
    # make output of attentions possible
    # bert_config.output_attentions=True
    print(bert_config)
    
    # load weights from checkpoint file created with tf.train.Checkpoint() and checkpoint.save()
    model = TFBertForSequenceClassification(config=bert_config)
    

    # update input vector size
    args.vector_size = args.config_dict['max_position_embeddings']

    # get list of testing tfrecords and number of reads per tfrecords
    test_file = sorted(glob.glob(os.path.join(args.tfrecords, '*.tfrec')))
    num_reads_file = sorted(glob.glob(os.path.join(args.tfrecords, '*-read_count')))

    with open(num_reads_file[0], 'r') as infile:
        num_reads = int(infile.readline())
    print(f'# sequences: {num_reads}')
    # compute number of steps required to iterate over entire test set
    test_steps = math.ceil(num_reads/(args.batch_size))

    # get id of reads
    with open(args.tsv_file, 'r') as f:
        content = f.readlines()
        reads_id = [line.rstrip().split('\t')[0].split('|')[2].split('-')[0] for line in content]
        classification_group = {line.rstrip().split('\t')[0].split('|')[2].split('-')[0]: line.rstrip().split('\t')[0].split('-')[1] for line in content}
        genomes_pos = {line.rstrip().split('\t')[0].split('|')[2].split('-')[0]: '-'.join(line.rstrip().split('\t')[0].split('-')[2:]) for line in content}
        reads_seq = {line.rstrip().split('\t')[0].split('|')[2].split('-')[0]: line.rstrip().split('\t')[1] for line in content}


    args.datatype = 'finetuning'
    test_input = build_dataset(args, test_file, num_labels, is_training=False, drop_remainder=False)

    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # set color palette
    heatmap_palette = sn.color_palette("viridis", as_cmap=True)

    data_to_plot = defaultdict(list)
    attentions_df = defaultdict(list)

    print(len(reads_id), test_steps)
    for batch, data in enumerate(test_input.take(test_steps), 0):
        if reads_id[batch] in [args.tp_read, args.fn_read]:
            outputs, pred_labels, pred_probs = get_attentions(data, model, test_accuracy)
            # get attentions weights from the 12 attention heads in each of the 12 attention layers
            attentions = list(outputs[-1])
            # print number of attention layers
            # print(len(attentions))
            # print dimensions of the output of the last attention layer
            # print(attentions[-1].shape)
            # shape of the attentions output: (batch_size, num_attention_head, max_position_embeddings, max_position_embeddings)
            # shape of the last attention head output: (max_position_embeddings, max_position_embeddings)

            print(reads_id[batch])
            print(reads_seq[reads_id[batch]], len(reads_seq[reads_id[batch]]))
            for i in range(len(data["input_ids"])):
                label = data["labels"][i].numpy()
                seq_ids = data["input_ids"][i].numpy()
                print(seq_ids)
                tokens = [vocab[i] for i in seq_ids]
                print(tokens)
                assert '[UKN]' not in tokens
                # reconstruct original sequence
                dna_seq = tokens[1]
                for j in range(2, len(tokens), 1):
                    if tokens[j] not in ['[PAD]', '[SEP]', '[UNK]']:
                        dna_seq += tokens[j][-1]
                print(dna_seq)
                assert dna_seq == reads_seq[reads_id[batch]]
                print(tokens)
                print(len(tokens))
                # get attention weights of the last attention head in the last attention layer for the sequence investigated, shape is (max_position_embeddings, max_position_embeddings)
                attentions_weights = attentions[-1][-1][i].numpy()
                df = pd.DataFrame(attentions_weights)
                print(df.shape)
                print(df.columns.tolist())
                df.columns = tokens
                # remove rows ['PAD'], ['CLS'] and ['SEP']
                idx_to_rm = [idx for idx in range(len(tokens)) if tokens[idx] in ['[PAD]', '[CLS]', '[SEP]']]
                df = df.drop(idx_to_rm, axis='index')
                # remove columns ['PAD'], ['CLS'] and ['SEP']
                df = df.drop('[PAD]', axis='columns')
                df = df.drop('[CLS]', axis='columns')
                df = df.drop('[SEP]', axis='columns')
                # get list of kmers in the sequence
                df_kmers = df.columns.tolist()
                # rename index to kmers
                df.index = df_kmers
                print(df)

                # get sum of attention weights by column
                df_sum = df.sum(axis=0).tolist()
                # # get mean of attention weights by column
                # df_mean = df.mean(axis=0).tolist()
                # # get max value of attention weights by column
                # df_max = df.max(axis=0).tolist()
                
                attentions_df[reads_id[batch]] = df

                df.to_csv(os.path.join(args.output_dir, f'{reads_id[batch]}_attention_map.csv'), index=False)
                            
                # sort dictionary based on values
                dict_kmers_sum = dict(zip(df_kmers, df_sum))
                dict_kmers_sum_sorted = dict(sorted(dict_kmers_sum.items(), key=lambda item: item[1], reverse=True))
                with open(os.path.join(args.output_dir, f'kmers_{len(df)}_{reads_id[batch]}.tsv'), 'w') as f:
                    for kmer, kmer_sum in dict_kmers_sum_sorted.items():
                        f.write(f'{kmer}\t{kmer_sum}\n')

                print(np.mean(list(dict_kmers_sum.values())), np.median(list(dict_kmers_sum.values())), min(list(dict_kmers_sum.values())), max(list(dict_kmers_sum.values())))

                df_values = df.values.flatten().tolist()
                data_to_plot[reads_id[batch]] = df_values
                # get kmers with high attention weights
                # filtered_df = df[['col1', 'col3']]
                # filtered_df = df.loc[:, (df >= np.mean(df.values.tolist())).any()]
                # print(f'cutoff value for attentions: {np.mean(df.values.tolist())}')
                # print('filtered_df')
                # print(filtered_df)
                # print(df.shape)
                # print(filtered_df.shape)
                # plot heatmap of attention weights
                plt.figure(figsize=(15, 15))
                if df.shape[0] < 50:
                    sn.heatmap(data=df, annot=False, xticklabels=df.columns, yticklabels=df.columns, cmap=heatmap_palette) 
                else:
                    sn.heatmap(data=df, annot=False, xticklabels=False, yticklabels=False, cmap=heatmap_palette) 
                plt.savefig(os.path.join(args.output_dir, f'attention_weights_heatmap_{len(df)}_{reads_id[batch]}.png'))
                plt.close()

        
            # if label == 0:
            #     attention_weights_label_0.append(df.values.flatten().tolist())
            #     # kmers_label_0 += filtered_df.columns.tolist()
            #     confidence_scores_label_0.append(pred_probs[i])
            #     # labels_0.append(label)
            #     if pred_labels[i] == label:
            #         predictions_label_0.append('c') 
            #     else:
            #         predictions_label_0.append('i')
            # else:
            #     attention_weights_label_1.append(df.values.flatten().tolist())
            #     # kmers_label_1 += filtered_df.columns.tolist()
            #     confidence_scores_label_1.append(pred_probs[i])
            #     if pred_labels[i] == label:
            #         predictions_label_1.append('c') 
            #     else:
            #         predictions_label_1.append('i') 
    
    # plot histograms of attention weights
    hist_palette = sn.color_palette("husl", len(data_to_plot))
    plt.figure(figsize=(10, 10))
    for idx, (key, value) in enumerate(data_to_plot.items(),0):
        sn.histplot(data=value, color=hist_palette[idx], alpha=0.5, kde=True, label=f'{key}-{classification_group[key]}-{genomes_pos[key]}-{len(reads_seq[key])}')
    plt.xlabel('Attention scores')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, f'attention_weights_hist.png'))
    plt.close()

    # get kmers inside matching and non matching regions between the FN read and the TP read(s)
    fn_genome_pos_start = min([int(genomes_pos[args.fn_read].split('-')[0]), int(genomes_pos[args.fn_read].split('-')[1])])
    fn_genome_pos_end = max([int(genomes_pos[args.fn_read].split('-')[0]), int(genomes_pos[args.fn_read].split('-')[1])])
    start_genome_pos = {args.fn_read : fn_genome_pos_start}
    end_genome_pos = {args.fn_read : fn_genome_pos_end}
    non_matching_seq = defaultdict(list)
    
    tp_genome_pos_start = min([int(genomes_pos[args.tp_read].split('-')[0]), int(genomes_pos[args.tp_read].split('-')[1])])
    tp_genome_pos_end = max([int(genomes_pos[args.tp_read].split('-')[0]), int(genomes_pos[args.tp_read].split('-')[1])])
    assert tp_genome_pos_end-tp_genome_pos_start+1 == len(reads_seq[args.tp_read]), f'{tp_genome_pos_end-tp_genome_pos_start}-{len(reads_seq[args.tp_read])}'
    assert fn_genome_pos_end-fn_genome_pos_start+1 == len(reads_seq[args.fn_read]), f'{fn_genome_pos_end-fn_genome_pos_start}-{len(reads_seq[args.fn_read])}'
    start_genome_pos[args.tp_read] = tp_genome_pos_start 
    end_genome_pos[args.tp_read] = tp_genome_pos_end
    print(tp_genome_pos_start, tp_genome_pos_end, fn_genome_pos_start, fn_genome_pos_end)
    tp_pos = list(range(tp_genome_pos_start, tp_genome_pos_end+1, 1))
    fn_pos = list(range(fn_genome_pos_start, fn_genome_pos_end+1, 1))
    matching_pos = [min(set(tp_pos).intersection(set(fn_pos))), max(set(tp_pos).intersection(set(fn_pos)))]
    
    tp_matching_seq = ''
    tp_non_matching_seq = ''
    genome_pos = tp_genome_pos_start
    for i in range(len(reads_seq[args.tp_read])):
        if genome_pos >= matching_pos[0] and genome_pos <= matching_pos[1]:
            tp_matching_seq += reads_seq[args.tp_read][i]
        if genome_pos <= matching_pos[0] or genome_pos >= matching_pos[1]:
            tp_non_matching_seq += reads_seq[args.tp_read][i]
        genome_pos += 1

    fn_matching_seq = ''
    fn_non_matching_seq = ''
    genome_pos = fn_genome_pos_start
    for i in range(len(reads_seq[args.fn_read])):
        if genome_pos >= matching_pos[0] and genome_pos <= matching_pos[1]:
            fn_matching_seq += reads_seq[args.fn_read][i]
        if genome_pos <= matching_pos[0] or genome_pos >= matching_pos[1]:
            fn_non_matching_seq += reads_seq[args.fn_read][i]
        genome_pos += 1

    assert fn_matching_seq == tp_matching_seq, f'{fn_matching_seq} - {tp_matching_seq}'

    with open(os.path.join(args.output_dir, f'{args.tp_read}_{args.fn_read}_matching_seq'), 'w') as f:
        f.write(f'matching positions: {matching_pos[0]}\t{matching_pos[1]}\n')
        f.write(f'tp start: {tp_genome_pos_start}\ttp end: {tp_genome_pos_end}\n')
        f.write(f'tp seq: {reads_seq[args.tp_read]}\n')
        f.write(f'tp non matching seq: {tp_non_matching_seq}\n')
        f.write(f'tp matching seq: {tp_matching_seq}\n')
        f.write(f'fn start: {fn_genome_pos_start}\tfn end: {fn_genome_pos_end}\n')
        f.write(f'fn seq: {reads_seq[args.fn_read]}\n')
        f.write(f'fn non matching seq: {fn_non_matching_seq}\n')
        f.write(f'fn matching seq: {fn_matching_seq}\n')

    # plot TP and FN along with sum of attention scores
    att_scores_out = open(os.path.join(args.output_dir, f'attention_scores_stats_{args.cutoff}.tsv'), 'w')
    strand = 1
    # gv = GenomeViz()
    # get length of segment to plot
    start_x_value = min(start_genome_pos.values())
    end_x_value = max(end_genome_pos.values())
    genome_pos_to_segment = {pos:idx for idx, pos in enumerate(range(start_x_value, end_x_value+1, 1), 0)}
    print(f'length of fragment shown: {end_x_value-start_x_value}\t{end_x_value}\t{start_x_value}')
    # gv.set_scale_xticks()
    non_matching_pos = defaultdict(list)

    # # add track for FN
    # fn_track_all = gv.add_feature_track(f'FN - key', end_x_value-start_x_value)
    # fn_track_non_match = gv.add_feature_track(f'FN - query', end_x_value-start_x_value)
    # # fn_track.add_subtrack(name='attentions', ylim=(0, max_y_value))
    # # add matching and non matching sequences with TP reads
    # fn_track_all.add_feature(genome_pos_to_segment[matching_pos[0]], genome_pos_to_segment[matching_pos[1]], strand, plotstyle="bigrbox", fc='blue')
    # right_non_matching_regions = []
    # left_non_matching_regions = []
    # for i in range(fn_genome_pos_start, fn_genome_pos_end+1, 1):
    #     if i > matching_pos[1]:
    #         right_non_matching_regions.append(i)
    #     if i < matching_pos[0]:
    #         left_non_matching_regions.append(i)

    # if len(right_non_matching_regions) != 0:
    #     non_matching_pos[args.fn_read] += right_non_matching_regions
    #     print(f'FN - right non matching positions: {genome_pos_to_segment[min(right_non_matching_regions)]}\t{genome_pos_to_segment[max(right_non_matching_regions)]}')
    #     fn_track_all.add_feature(genome_pos_to_segment[min(right_non_matching_regions)], genome_pos_to_segment[max(right_non_matching_regions)], strand, plotstyle="bigrbox", fc='black')
    #     fn_track_non_match.add_feature(genome_pos_to_segment[min(right_non_matching_regions)], genome_pos_to_segment[max(right_non_matching_regions)], strand, plotstyle="bigrbox", fc='black')
    # if len(left_non_matching_regions) != 0:
    #     non_matching_pos[args.fn_read] += left_non_matching_regions
    #     print(f'FN - left non matching positions: {genome_pos_to_segment[min(left_non_matching_regions)]}\t{genome_pos_to_segment[max(left_non_matching_regions)]}')
    #     fn_track_all.add_feature(genome_pos_to_segment[min(left_non_matching_regions)], genome_pos_to_segment[max(left_non_matching_regions)], strand, plotstyle="bigrbox", fc='black')
    #     fn_track_non_match.add_feature(genome_pos_to_segment[min(left_non_matching_regions)], genome_pos_to_segment[max(left_non_matching_regions)], strand, plotstyle="bigrbox", fc='black')


    # # normalize attention scores
    # min_attention_score = min(attentions_df[args.fn_read].values.flatten().tolist())
    # max_attention_scores = max(attentions_df[args.fn_read].values.flatten().tolist())
    # print(f'min attention score: {min_attention_score}')
    # print(f'max attention score: {max_attention_scores}')
    # list_attention_scores  = attentions_df[args.fn_read].values.flatten().tolist()
    # att_scores_out.write(f'FN - before normalization\nmean\t{statistics.mean(list_attention_scores)}\nmedian\t{statistics.median(list_attention_scores)}\nmin\t{min(list_attention_scores)}\nmax\t{max(list_attention_scores)}\n')
    # attentions_df[args.fn_read] = attentions_df[args.fn_read].applymap(lambda x: Normalize(x, x_min=min_attention_score, x_max=max_attention_scores))
    # list_attention_scores  = attentions_df[args.fn_read].values.flatten().tolist()
    # att_scores_out.write(f'FN - after normalization\nmean\t{statistics.mean(list_attention_scores)}\nmedian\t{statistics.median(list_attention_scores)}\nmin\t{min(list_attention_scores)}\nmax\t{max(list_attention_scores)}\n')

    # # normalize values in dataframes
    # color = 'red'
    # for idx, track in enumerate(gv.feature_tracks, 0):
    #     if idx == 1:
    #         print(track)
    #         # subtrack = track.get_subtrack('attentions')
    #         read_id = args.fn_read
    #         print(read_id, classification_group[read_id])
    #         print(attentions_df[read_id])
    #         # get attentions with all kmers in sequence for each kmer in the non matching sequence
    #         for query_read_pos, i in enumerate(range(start_genome_pos[read_id], end_genome_pos[read_id]-4+1, 1), 0):
    #             # check if position is in a non-matching region
    #             if i in non_matching_pos[read_id]:
    #                 # get position of first and last nucleotide in the kmer
    #                 query_first_pos = query_read_pos
    #                 query_last_pos = query_read_pos + 4
    #                 query_first_genome_pos = genome_pos_to_segment[i]
    #                 query_last_genome_pos = genome_pos_to_segment[i+4]
    #                 query_kmer = reads_seq[read_id][query_first_pos:query_last_pos]
    #                 for key_read_pos, j in enumerate(range(start_genome_pos[read_id], end_genome_pos[read_id]-4+1, 1), 0):
    #                     key_first_pos = key_read_pos
    #                     key_last_pos = key_read_pos + 4
    #                     key_first_genome_pos = genome_pos_to_segment[j]
    #                     key_last_genome_pos = genome_pos_to_segment[j+4]
    #                     key_kmer = reads_seq[read_id][key_first_pos:key_last_pos]
    #                     attention_score = attentions_df[read_id].iloc[query_first_pos, key_first_pos]
    #                     if attention_score > args.cutoff:
    #                         if classification_group[read_id] == 'fn':
    #                             query_info = (f'FN - query', query_first_genome_pos, query_last_genome_pos)
    #                             key_info = (f'FN - key', key_first_genome_pos, key_last_genome_pos)
    #                         elif classification_group[read_id] == 'tp':
    #                             query_info = (f'TP - query', query_first_pos, query_last_genome_pos)
    #                             key_info = (f'TP - key', key_first_genome_pos, key_last_genome_pos)
    #                         fn_query_key_out.write(f'{query_kmer}\t{query_first_pos}\t{query_first_genome_pos}\t{query_last_pos}\t{query_last_genome_pos}\t{key_kmer}\t{key_first_pos}\t{key_first_genome_pos}\t{key_last_pos}\t{key_last_genome_pos}\t{attention_score}\n')
    #                         gv.add_link(query_info, key_info, color=color, v=attention_score, vmin=0.0, curve=True)
    #         gv.set_colorbar([color], vmin=0.0)

    # fig = gv.plotfig()
    # fig.savefig(os.path.join(args.output_dir, f'plot_fn_{args.fn_read}_{args.cutoff}.png'), dpi=300)


    # # add tracks for TP + matching and non matching sequences with FN read
    # tp_query_key_out = open(os.path.join(args.output_dir, f'tp_query_key_{args.cutoff}.tsv'), 'w')
    # # gv = GenomeViz()
    # color_non_matching = 'darkviolet'
    # color_matching = 'blue'
    # # gv.set_scale_xticks()
    # print(f'TP - matching positions: {genome_pos_to_segment[matching_pos[0]]}\t{genome_pos_to_segment[matching_pos[1]]}')
    # tp_track_key = gv.add_feature_track(f'TP - key', end_x_value-start_x_value)
    # tp_track_query = gv.add_feature_track(f'TP - query', end_x_value-start_x_value)
    # tp_track_key.add_feature(genome_pos_to_segment[matching_pos[0]], genome_pos_to_segment[matching_pos[1]], strand, plotstyle="bigrbox", fc=color_matching)
    # tp_track_query.add_feature(genome_pos_to_segment[matching_pos[0]], genome_pos_to_segment[matching_pos[1]], strand, plotstyle="bigrbox", fc=color_matching)

    # right_non_matching_regions = []
    # left_non_matching_regions = []
    # for i in range(start_genome_pos[args.tp_read], end_genome_pos[args.tp_read]+1, 1):
    #     if i > matching_pos[1]:
    #         right_non_matching_regions.append(i)
    #     if i < matching_pos[0]:
    #         left_non_matching_regions.append(i)

    # if len(right_non_matching_regions) != 0:
    #     non_matching_pos[args.tp_read] += right_non_matching_regions
    #     print(f'TP - right non matching positions: {genome_pos_to_segment[min(right_non_matching_regions)]}\t{genome_pos_to_segment[max(right_non_matching_regions)]}')
    #     tp_track_query.add_feature(genome_pos_to_segment[min(right_non_matching_regions)], genome_pos_to_segment[max(right_non_matching_regions)], strand, plotstyle="bigrbox", fc=color_non_matching)
    #     tp_track_key.add_feature(genome_pos_to_segment[min(right_non_matching_regions)], genome_pos_to_segment[max(right_non_matching_regions)], strand, plotstyle="bigrbox", fc=color_non_matching)

    # if len(left_non_matching_regions) != 0:
    #     non_matching_pos[args.tp_read] += left_non_matching_regions
    #     print(f'TP - left non matching positions: {genome_pos_to_segment[min(left_non_matching_regions)]}\t{genome_pos_to_segment[max(left_non_matching_regions)]}')
    #     tp_track_query.add_feature(genome_pos_to_segment[min(left_non_matching_regions)], genome_pos_to_segment[max(left_non_matching_regions)], strand, plotstyle="bigrbox", fc=color_non_matching)
    #     tp_track_key.add_feature(genome_pos_to_segment[min(left_non_matching_regions)], genome_pos_to_segment[max(left_non_matching_regions)], strand, plotstyle="bigrbox", fc=color_non_matching)



    # # normalize attention scores
    # min_attention_score = min(attentions_df[args.tp_read].values.flatten().tolist())
    # max_attention_scores = max(attentions_df[args.tp_read].values.flatten().tolist())
    # print(f'min attention score: {min_attention_score}')
    # print(f'max attention score: {max_attention_scores}')
    # list_attention_scores  = attentions_df[args.tp_read].values.flatten().tolist()
    # att_scores_out.write(f'\nTP - before normalization\nmean\t{statistics.mean(list_attention_scores)}\nmedian\t{statistics.median(list_attention_scores)}\nmin\t{min(list_attention_scores)}\nmax\t{max(list_attention_scores)}\n')
    # attentions_df[args.tp_read] = attentions_df[args.tp_read].applymap(lambda x: Normalize(x, x_min=min_attention_score, x_max=max_attention_scores))
    # list_attention_scores  = attentions_df[args.tp_read].values.flatten().tolist()
    # att_scores_out.write(f'TP - after normalization\nmean\t{statistics.mean(list_attention_scores)}\nmedian\t{statistics.median(list_attention_scores)}\nmin\t{min(list_attention_scores)}\nmax\t{max(list_attention_scores)}\n')

    # for idx, track in enumerate(gv.feature_tracks, 0):
    #     if idx == 1:
    #         print(track)
    #         # subtrack = track.get_subtrack('attentions')
    #         read_id = args.tp_read
    #         print(read_id, classification_group[read_id])
    #         print(attentions_df[read_id])
    #         # get attentions with all kmers in sequence for each kmer in the non matching sequence
    #         for query_read_pos, i in enumerate(range(start_genome_pos[read_id], end_genome_pos[read_id]-4+1, 1), 0):
    #             # get position of first and last nucleotide in the kmer
    #             query_first_pos = query_read_pos
    #             query_last_pos = query_read_pos + 4
    #             query_first_genome_pos = genome_pos_to_segment[i]
    #             query_last_genome_pos = genome_pos_to_segment[i+4]
    #             query_kmer = reads_seq[read_id][query_first_pos:query_last_pos]
    #             for key_read_pos, j in enumerate(range(start_genome_pos[read_id], end_genome_pos[read_id]-4+1, 1), 0):
    #                 key_first_pos = key_read_pos
    #                 key_last_pos = key_read_pos + 4
    #                 key_first_genome_pos = genome_pos_to_segment[j]
    #                 key_last_genome_pos = genome_pos_to_segment[j+4]
    #                 key_kmer = reads_seq[read_id][key_first_pos:key_last_pos]
    #                 attention_score = attentions_df[read_id].iloc[query_first_pos, key_first_pos]
    #                 if attention_score > args.cutoff:
    #                     if classification_group[read_id] == 'fn':
    #                         query_info = (f'FN - query', query_first_genome_pos, query_last_genome_pos)
    #                         key_info = (f'FN - key', key_first_genome_pos, key_last_genome_pos)
    #                     elif classification_group[read_id] == 'tp':
    #                         query_info = (f'TP - query', query_first_genome_pos, query_last_genome_pos)
    #                         key_info = (f'TP - key', key_first_genome_pos, key_last_genome_pos)
    #                     # check if position is in a non-matching region
    #                     if i in non_matching_pos[read_id]:
    #                         tp_query_key_out.write(f'non-matching\t{query_kmer}\t{query_first_pos}\t{query_first_genome_pos}\t{query_last_pos}\t{query_last_genome_pos}\t{key_kmer}\t{key_first_pos}\t{key_first_genome_pos}\t{key_last_pos}\t{key_last_genome_pos}\t{attention_score}\n')
    #                         gv.add_link(query_info, key_info, color=color_non_matching, v=attention_score, vmin=0.0, curve=True)
    #                     elif i >= matching_pos[0] and i <= matching_pos[1]:
    #                         tp_query_key_out.write(f'matching\t{query_kmer}\t{query_first_pos}\t{query_first_genome_pos}\t{query_last_pos}\t{query_last_genome_pos}\t{key_kmer}\t{key_first_pos}\t{key_first_genome_pos}\t{key_last_pos}\t{key_last_genome_pos}\t{attention_score}\n')
    #                         gv.add_link(query_info, key_info, color=color_matching, v=attention_score, vmin=0.0, curve=True)

    #         gv.set_colorbar([color_matching, color_non_matching], vmin=0.0)
            
    # fig = gv.plotfig()
    # fig.savefig(os.path.join(args.output_dir, f'plot_tp_{args.tp_read}_{args.cutoff}.png'), dpi=300)


    # add tracks for TP + matching and non matching sequences with FN read
    gv = GenomeViz()
    fn_color = 'darkviolet'
    tp_color = 'blue'
    color_matching = 'black'
    gv.set_scale_xticks()
    # print(f'TP - matching positions: {genome_pos_to_segment[matching_pos[0]]}\t{genome_pos_to_segment[matching_pos[1]]}')
    track_key = gv.add_feature_track(f'key', end_x_value-start_x_value)
    track_query = gv.add_feature_track(f'query', end_x_value-start_x_value)
    # add matching sequence 
    track_key.add_feature(genome_pos_to_segment[matching_pos[0]], genome_pos_to_segment[matching_pos[1]], strand, plotstyle="bigrbox", fc=color_matching, label="identical", text_kws=dict(rotation=0, hpos="center"))
    track_query.add_feature(genome_pos_to_segment[matching_pos[0]], genome_pos_to_segment[matching_pos[1]], strand, plotstyle="bigrbox", fc=color_matching)

    # find non-matching sequences of TP read
    right_non_matching_regions = []
    left_non_matching_regions = []
    for i in range(start_genome_pos[args.tp_read], end_genome_pos[args.tp_read]+1, 1):
        if i > matching_pos[1]:
            right_non_matching_regions.append(i)
        if i < matching_pos[0]:
            left_non_matching_regions.append(i)

    if len(right_non_matching_regions) != 0:
        non_matching_pos[args.tp_read] += right_non_matching_regions
        print(f'TP - right non matching positions: {genome_pos_to_segment[min(right_non_matching_regions)]}\t{genome_pos_to_segment[max(right_non_matching_regions)]}')
        track_query.add_feature(genome_pos_to_segment[min(right_non_matching_regions)], genome_pos_to_segment[max(right_non_matching_regions)], strand, plotstyle="bigrbox", fc=tp_color)
        track_key.add_feature(genome_pos_to_segment[min(right_non_matching_regions)], genome_pos_to_segment[max(right_non_matching_regions)], strand, plotstyle="bigrbox", fc=tp_color, label="true positive", text_kws=dict(rotation=0, hpos="center", color=tp_color))

    if len(left_non_matching_regions) != 0:
        non_matching_pos[args.tp_read] += left_non_matching_regions
        print(f'TP - left non matching positions: {genome_pos_to_segment[min(left_non_matching_regions)]}\t{genome_pos_to_segment[max(left_non_matching_regions)]}')
        track_query.add_feature(genome_pos_to_segment[min(left_non_matching_regions)], genome_pos_to_segment[max(left_non_matching_regions)], strand, plotstyle="bigrbox", fc=tp_color)
        track_key.add_feature(genome_pos_to_segment[min(left_non_matching_regions)], genome_pos_to_segment[max(left_non_matching_regions)], strand, plotstyle="bigrbox", fc=tp_color, label="true positive", text_kws=dict(rotation=0, hpos="center", color=tp_color))

    # find non-matching sequences of FN read
    right_non_matching_regions = []
    left_non_matching_regions = []
    for i in range(fn_genome_pos_start, fn_genome_pos_end+1, 1):
        if i > matching_pos[1]:
            right_non_matching_regions.append(i)
        if i < matching_pos[0]:
            left_non_matching_regions.append(i)

    if len(right_non_matching_regions) != 0:
        non_matching_pos[args.fn_read] += right_non_matching_regions
        print(f'FN - right non matching positions: {genome_pos_to_segment[min(right_non_matching_regions)]}\t{genome_pos_to_segment[max(right_non_matching_regions)]}')
        track_query.add_feature(genome_pos_to_segment[min(right_non_matching_regions)], genome_pos_to_segment[max(right_non_matching_regions)], strand, plotstyle="bigrbox", fc=fn_color)
        track_key.add_feature(genome_pos_to_segment[min(right_non_matching_regions)], genome_pos_to_segment[max(right_non_matching_regions)], strand, plotstyle="bigrbox", fc=fn_color, label="false negative", text_kws=dict(rotation=0, hpos="center", color=fn_color))
    if len(left_non_matching_regions) != 0:
        non_matching_pos[args.fn_read] += left_non_matching_regions
        print(f'FN - left non matching positions: {genome_pos_to_segment[min(left_non_matching_regions)]}\t{genome_pos_to_segment[max(left_non_matching_regions)]}')
        track_query.add_feature(genome_pos_to_segment[min(left_non_matching_regions)], genome_pos_to_segment[max(left_non_matching_regions)], strand, plotstyle="bigrbox", fc=fn_color)
        track_key.add_feature(genome_pos_to_segment[min(left_non_matching_regions)], genome_pos_to_segment[max(left_non_matching_regions)], strand, plotstyle="bigrbox", fc=fn_color, label="false negative", text_kws=dict(rotation=0, hpos="center", color=fn_color))


    # add attention scores info for TP
    # normalize attention scores
    min_attention_score = min(attentions_df[args.tp_read].values.flatten().tolist())
    max_attention_scores = max(attentions_df[args.tp_read].values.flatten().tolist())
    print(f'min attention score: {min_attention_score}')
    print(f'max attention score: {max_attention_scores}')
    list_attention_scores  = attentions_df[args.tp_read].values.flatten().tolist()
    att_scores_out.write(f'\nTP - before normalization\nmean\t{statistics.mean(list_attention_scores)}\nmedian\t{statistics.median(list_attention_scores)}\nmin\t{min(list_attention_scores)}\nmax\t{max(list_attention_scores)}\n')
    attentions_df[args.tp_read] = attentions_df[args.tp_read].applymap(lambda x: Normalize(x, x_min=min_attention_score, x_max=max_attention_scores))
    list_attention_scores  = attentions_df[args.tp_read].values.flatten().tolist()
    att_scores_out.write(f'TP - after normalization\nmean\t{statistics.mean(list_attention_scores)}\nmedian\t{statistics.median(list_attention_scores)}\nmin\t{min(list_attention_scores)}\nmax\t{max(list_attention_scores)}\n')
    tp_query_key_out = open(os.path.join(args.output_dir, f'tp_query_key_{args.cutoff}.tsv'), 'w')
    for idx, track in enumerate(gv.feature_tracks, 0):
        if idx == 1:
            print(track)
            # subtrack = track.get_subtrack('attentions')
            read_id = args.tp_read
            print(read_id, classification_group[read_id])
            print(attentions_df[read_id])
            # get attentions with all kmers in sequence for each kmer in the non matching sequence
            for query_read_pos, i in enumerate(range(start_genome_pos[read_id], end_genome_pos[read_id]-4+1, 1), 0):
                # get position of first and last nucleotide in the kmer
                query_first_pos = query_read_pos
                query_last_pos = query_read_pos + 4
                query_first_genome_pos = genome_pos_to_segment[i]
                query_last_genome_pos = genome_pos_to_segment[i+4]
                query_kmer = reads_seq[read_id][query_first_pos:query_last_pos]
                for key_read_pos, j in enumerate(range(start_genome_pos[read_id], end_genome_pos[read_id]-4+1, 1), 0):
                    key_first_pos = key_read_pos
                    key_last_pos = key_read_pos + 4
                    key_first_genome_pos = genome_pos_to_segment[j]
                    key_last_genome_pos = genome_pos_to_segment[j+4]
                    key_kmer = reads_seq[read_id][key_first_pos:key_last_pos]
                    attention_score = attentions_df[read_id].iloc[query_first_pos, key_first_pos]
                    if attention_score > args.cutoff:
                        query_info = (f'query', query_first_genome_pos, query_last_genome_pos)
                        key_info = (f'key', key_first_genome_pos, key_last_genome_pos)
                        # check if position is in a non-matching region
                        if i in non_matching_pos[read_id]:
                            tp_query_key_out.write(f'non-matching\t{query_kmer}\t{query_first_pos}\t{query_first_genome_pos}\t{query_last_pos}\t{query_last_genome_pos}\t{key_kmer}\t{key_first_pos}\t{key_first_genome_pos}\t{key_last_pos}\t{key_last_genome_pos}\t{attention_score}\n')
                            gv.add_link(query_info, key_info, color=tp_color, v=attention_score, vmin=0.0, curve=True)
                        elif i >= matching_pos[0] and i <= matching_pos[1]:
                            tp_query_key_out.write(f'matching\t{query_kmer}\t{query_first_pos}\t{query_first_genome_pos}\t{query_last_pos}\t{query_last_genome_pos}\t{key_kmer}\t{key_first_pos}\t{key_first_genome_pos}\t{key_last_pos}\t{key_last_genome_pos}\t{attention_score}\n')
                            gv.add_link(query_info, key_info, color=tp_color, v=attention_score, vmin=0.0, curve=True)

    # add attention scores info for FN
    # normalize attention scores
    min_attention_score = min(attentions_df[args.fn_read].values.flatten().tolist())
    max_attention_scores = max(attentions_df[args.fn_read].values.flatten().tolist())
    print(f'min attention score: {min_attention_score}')
    print(f'max attention score: {max_attention_scores}')
    list_attention_scores  = attentions_df[args.fn_read].values.flatten().tolist()
    att_scores_out.write(f'\nFN - before normalization\nmean\t{statistics.mean(list_attention_scores)}\nmedian\t{statistics.median(list_attention_scores)}\nmin\t{min(list_attention_scores)}\nmax\t{max(list_attention_scores)}\n')
    attentions_df[args.fn_read] = attentions_df[args.fn_read].applymap(lambda x: Normalize(x, x_min=min_attention_score, x_max=max_attention_scores))
    list_attention_scores  = attentions_df[args.fn_read].values.flatten().tolist()
    att_scores_out.write(f'FN - after normalization\nmean\t{statistics.mean(list_attention_scores)}\nmedian\t{statistics.median(list_attention_scores)}\nmin\t{min(list_attention_scores)}\nmax\t{max(list_attention_scores)}\n')
    fn_query_key_out = open(os.path.join(args.output_dir, f'fn_query_key_{args.cutoff}.tsv'), 'w')
    for idx, track in enumerate(gv.feature_tracks, 0):
        if idx == 1:
            print(track)
            # subtrack = track.get_subtrack('attentions')
            read_id = args.fn_read
            print(read_id, classification_group[read_id])
            print(attentions_df[read_id])
            # get attentions with all kmers in sequence for each kmer in the non matching sequence
            for query_read_pos, i in enumerate(range(start_genome_pos[read_id], end_genome_pos[read_id]-4+1, 1), 0):
                # get position of first and last nucleotide in the kmer
                query_first_pos = query_read_pos
                query_last_pos = query_read_pos + 4
                query_first_genome_pos = genome_pos_to_segment[i]
                query_last_genome_pos = genome_pos_to_segment[i+4]
                query_kmer = reads_seq[read_id][query_first_pos:query_last_pos]
                for key_read_pos, j in enumerate(range(start_genome_pos[read_id], end_genome_pos[read_id]-4+1, 1), 0):
                    key_first_pos = key_read_pos
                    key_last_pos = key_read_pos + 4
                    key_first_genome_pos = genome_pos_to_segment[j]
                    key_last_genome_pos = genome_pos_to_segment[j+4]
                    key_kmer = reads_seq[read_id][key_first_pos:key_last_pos]
                    attention_score = attentions_df[read_id].iloc[query_first_pos, key_first_pos]
                    if attention_score > args.cutoff:
                        query_info = (f'query', query_first_genome_pos, query_last_genome_pos)
                        key_info = (f'key', key_first_genome_pos, key_last_genome_pos)
                        # check if position is in a non-matching region
                        if i in non_matching_pos[read_id]:
                            fn_query_key_out.write(f'non-matching\t{query_kmer}\t{query_first_pos}\t{query_first_genome_pos}\t{query_last_pos}\t{query_last_genome_pos}\t{key_kmer}\t{key_first_pos}\t{key_first_genome_pos}\t{key_last_pos}\t{key_last_genome_pos}\t{attention_score}\n')
                            gv.add_link(query_info, key_info, color=fn_color, v=attention_score, vmin=0.0, curve=True)
                        elif i >= matching_pos[0] and i <= matching_pos[1]:
                            fn_query_key_out.write(f'matching\t{query_kmer}\t{query_first_pos}\t{query_first_genome_pos}\t{query_last_pos}\t{query_last_genome_pos}\t{key_kmer}\t{key_first_pos}\t{key_first_genome_pos}\t{key_last_pos}\t{key_last_genome_pos}\t{attention_score}\n')
                            gv.add_link(query_info, key_info, color=fn_color, v=attention_score, vmin=0.0, curve=True)

    gv.set_colorbar([fn_color, tp_color], vmin=0.0)
            
    fig = gv.plotfig()
    fig.savefig(os.path.join(args.output_dir, f'plot_tp_{args.tp_read}_fn_{args.fn_read}_{args.cutoff}.png'), dpi=300)




if __name__ == "__main__":
    main()
