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
import matplotlib.pyplot as plt 
from collections import defaultdict
import random


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
    outputs = model(**data)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords', type=str, help='path to tfrecords', required=True)
    parser.add_argument('--tsv_file', type=str, help='path to tsv file', required=True)
    parser.add_argument('--output_dir', type=str, help='directory to store results', default=os.getcwd())
    parser.add_argument('--init_lr', type=float, help='initial learning rate', default=0.0001)
    parser.add_argument('--batch_size', type=int, help='batch size per gpu', default=8192)
    parser.add_argument('--num_labels', type=int, help='number of labels', default=2)
    parser.add_argument('--k_value', type=int, help='length of kmer strings', default=12)
    parser.add_argument('--list_reads_id', nargs='+', help='list of reads id to analyze', required=True)
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
    bert_config.output_attentions=True
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

    print(len(reads_id), test_steps)
    print(f'list of reads: {args.list_reads_id}\t{len(args.list_reads_id)}')
    for batch, data in enumerate(test_input.take(test_steps), 0):
        if reads_id[batch] in args.list_reads_id:
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
                seq_kmers = [vocab[i] for i in seq_ids]
                print(seq_kmers)
                # reconstruct original sequence
                dna_seq = seq_kmers[1]
                for j in range(2, len(seq_kmers), 1):
                    if seq_kmers[j] not in ['[PAD]', '[SEP]', '[UNK]']:
                        dna_seq += seq_kmers[j][-1]
                print(dna_seq)
                assert dna_seq == reads_seq[reads_id[batch]]
                print(seq_kmers)
                print(len(seq_kmers))
                # get attention weights of the last attention head in the last attention layer for the sequence investigated, shape is (max_position_embeddings, max_position_embeddings)
                attentions_weights = attentions[-1][-1][i].numpy()
                df = pd.DataFrame(attentions_weights)
                print(df.shape)
                df.columns = seq_kmers
                # remove rows ['PAD']
                pad_idx = [idx for idx in range(len(seq_kmers)) if seq_kmers[idx] == '[PAD]']
                df = df.drop(pad_idx, axis='index')
                # remove columns ['PAD']
                df = df.drop('[PAD]', axis='columns')
                print(df)
                
                # get sum of attention weights by column
                df_sum = df.sum(axis=0).tolist()
                df_kmers = df.columns.tolist()
                            
                # sort dictionary based on values
                dict_kmers_sum = dict(zip(df_kmers, df_sum))
                dict_kmers_sum_sorted = dict(sorted(dict_kmers_sum.items(), key=lambda item: item[1], reverse=True))
                with open(os.path.join(args.output_dir, f'kmers_{len(df)}_{reads_id[batch]}.tsv'), 'w') as f:
                    for kmer, kmer_sum in dict_kmers_sum_sorted.items():
                        f.write(f'{kmer}\t{kmer_sum}\n')

                print(np.mean(list(dict_kmers_sum.values())), np.median(list(dict_kmers_sum.values())), min(list(dict_kmers_sum.values())), max(list(dict_kmers_sum.values())))


                df_values = df.values.flatten().tolist()
                with open(os.path.join(args.output_dir, f'stats_att_{len(df)}_{reads_id[batch]}.tsv'), 'w') as f:
                    f.write(f'{np.mean(df_values)}\t{np.median(df_values)}\t{min(df_values)}\t{max(df_values)}')

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
    fn_read_id = [key for key, value in classification_group.items() if value == 'fn' and key in args.list_reads_id]
    tp_read_id = [key for key, value in classification_group.items() if value == 'tp' and key in args.list_reads_id]
    for tp_read in tp_read_id:
        print(tp_read, fn_read_id[0])
        tp_genome_pos_start = min([int(genomes_pos[tp_read].split('-')[0]), int(genomes_pos[tp_read].split('-')[1])])
        tp_genome_pos_end = max([int(genomes_pos[tp_read].split('-')[0]), int(genomes_pos[tp_read].split('-')[1])])
        assert tp_genome_pos_end-tp_genome_pos_start+1 == len(reads_seq[tp_read]), f'{tp_genome_pos_end-tp_genome_pos_start}-{len(reads_seq[tp_read])}'
        fn_genome_pos_start = min([int(genomes_pos[fn_read_id[0]].split('-')[0]), int(genomes_pos[fn_read_id[0]].split('-')[1])])
        fn_genome_pos_end = max([int(genomes_pos[fn_read_id[0]].split('-')[0]), int(genomes_pos[fn_read_id[0]].split('-')[1])])
        assert fn_genome_pos_end-fn_genome_pos_start+1 == len(reads_seq[fn_read_id[0]]), f'{fn_genome_pos_end-fn_genome_pos_start}-{len(reads_seq[fn_read_id[0]])}'
        print(tp_genome_pos_start, tp_genome_pos_end, fn_genome_pos_start, fn_genome_pos_end)
        tp_pos = list(range(tp_genome_pos_start, tp_genome_pos_end+1, 1))
        fn_pos = list(range(fn_genome_pos_start, fn_genome_pos_end+1, 1))
        overlap = [min(set(tp_pos).intersection(set(fn_pos))), max(set(tp_pos).intersection(set(fn_pos)))]
        print(overlap)
        tp_overlap_seq = ''
        tp_non_overlap_seq = ''
        genome_pos = tp_genome_pos_start
        for i in range(len(reads_seq[tp_read])):
            if genome_pos >= overlap[0] and genome_pos <= overlap[1]:
                tp_overlap_seq += reads_seq[tp_read][i]
            if genome_pos <= overlap[0] or genome_pos >= overlap[1]:
                tp_non_overlap_seq += reads_seq[tp_read][i]
            genome_pos += 1

        fn_overlap_seq = ''
        fn_non_overlap_seq = ''
        genome_pos = fn_genome_pos_start
        for i in range(len(reads_seq[fn_read_id[0]])):
            if genome_pos >= overlap[0] and genome_pos <= overlap[1]:
                fn_overlap_seq += reads_seq[fn_read_id[0]][i]
            if genome_pos <= overlap[0] or genome_pos >= overlap[1]:
                fn_non_overlap_seq += reads_seq[fn_read_id[0]][i]
            genome_pos += 1

        assert fn_overlap_seq == tp_overlap_seq
        with open(os.path.join(args.output_dir, f'overlap_seq'), 'w') as f:
            f.write(f'tp non overlap seq: {tp_non_overlap_seq}')
            f.write(f'tp seq: {reads_seq[tp_read]}')

            f.write(f'fn non overlap seq: {fn_non_overlap_seq}')
            f.write(f'fn seq: {reads_seq[fn_read_id[0]]}')



    # # plot histogram of attention weights for other labels
    # confidence_scores_label_0_correct = [confidence_scores_label_0[i] for i in range(len(confidence_scores_label_0)) if predictions_label_0[i] == 'c']
    # confidence_scores_label_0_incorrect = [confidence_scores_label_0[i] for i in range(len(confidence_scores_label_0)) if predictions_label_0[i] == 'i']
    # plt.figure(figsize=(10, 6))
    # sn.histplot(data=confidence_scores_label_0_correct)
    # plt.xlabel('Confidence Scores')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.savefig(os.path.join(args.output_dir, 'confidence_scores_correct_hist_other.png'))
    # plt.figure(figsize=(10, 6))
    # sn.histplot(data=confidence_scores_label_0_incorrect)
    # plt.xlabel('Confidence Scores')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.savefig(os.path.join(args.output_dir, 'confidence_scores_incorrect_hist_other.png'))

    # # plot histogram of attention weights for label investigated
    # confidence_scores_label_1_correct = [confidence_scores_label_1[i] for i in range(len(confidence_scores_label_1)) if predictions_label_1[i] == 'c']
    # confidence_scores_label_1_incorrect = [confidence_scores_label_1[i] for i in range(len(confidence_scores_label_1)) if predictions_label_1[i] == 'i']
    # plt.figure(figsize=(10, 6))
    # sn.histplot(data=confidence_scores_label_1_correct)
    # plt.xlabel('Confidence Scores')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.savefig(os.path.join(args.output_dir, 'confidence_scores_correct_hist_label.png'))
    # plt.figure(figsize=(10, 6))
    # sn.histplot(data=confidence_scores_label_1_incorrect)
    # plt.xlabel('Confidence Scores')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.savefig(os.path.join(args.output_dir, 'confidence_scores_incorrect_hist_label.png'))

    # # plot histogram of confidence scores for other labels
    # attention_weights_label_0_correct = [attention_weights_label_0[i] for i in range(len(attention_weights_label_0)) if predictions_label_0[i] == 'c']
    # attention_weights_label_0_incorrect = [attention_weights_label_0[i] for i in range(len(attention_weights_label_0)) if predictions_label_0[i] == 'i']
    # plt.figure(figsize=(10, 6))
    # sn.histplot(data=attention_weights_label_0_correct)
    # plt.xlabel('Attention Weights')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.savefig(os.path.join(args.output_dir, 'attention_weights_correct_hist_other.png'))
    # plt.figure(figsize=(10, 6))
    # sn.histplot(data=attention_weights_label_0_incorrect)
    # plt.xlabel('Attention Weights')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.savefig(os.path.join(args.output_dir, 'attention_weights_incorrect_hist_other.png'))

    # # plot histogram of attention weights for label investigated
    # attention_weights_label_1_correct = [attention_weights_label_1[i] for i in range(len(attention_weights_label_1)) if predictions_label_1[i] == 'c']
    # attention_weights_label_1_incorrect = [attention_weights_label_1[i] for i in range(len(attention_weights_label_1)) if predictions_label_1[i] == 'i']
    # plt.figure(figsize=(10, 6))
    # sn.histplot(data=attention_weights_label_1_correct)
    # plt.xlabel('Attention Weights')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.savefig(os.path.join(args.output_dir, 'attention_weights_correct_hist_label.png'))
    # plt.figure(figsize=(10, 6))
    # sn.histplot(data=attention_weights_label_1_incorrect)
    # plt.xlabel('Attention Weights')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.savefig(os.path.join(args.output_dir, 'attention_weights_incorrect_hist_label.png'))

    # # store list of relevant kmers
    # print(f'# relevant kmers for label 0: {len(set(kmers_label_0))}')
    # print(f'# relevant kmers for label 1: {len(set(kmers_label_1))}')
    # kmers_label_0_correct = set([kmers_label_0[i] for i in range(len(kmers_label_0)) if predictions_label_0[i] == 'c'])
    # kmers_label_0_incorrect = set([kmers_label_0[i] for i in range(len(kmers_label_0)) if predictions_label_0[i] == 'i'])
    # with open(os.path.join(args.output_dir, 'relevant_kmers_correct_other'), 'w') as f:
    #     f.write('\n'.join(list(kmers_label_0_correct)))
    # with open(os.path.join(args.output_dir, 'relevant_kmers_incorrect_other'), 'w') as f:
    #     f.write('\n'.join(list(kmers_label_0_incorrect)))

    # kmers_label_1_correct = set([kmers_label_1[i] for i in range(len(kmers_label_1)) if predictions_label_1[i] == 'c'])
    # kmers_label_1_incorrect = set([kmers_label_1[i] for i in range(len(kmers_label_1)) if predictions_label_1[i] == 'i'])
    # with open(os.path.join(args.output_dir, 'relevant_kmers_incorrect_label'), 'w') as f:
    #     f.write('\n'.join(list(kmers_label_1_correct)))
    # with open(os.path.join(args.output_dir, 'relevant_kmers_incorrect_label'), 'w') as f:
    #     f.write('\n'.join(list(kmers_label_1_incorrect)))

    # label_0_correct = set([labels_0[i] for i in range(len(labels_0)) if predictions_label_0[i] == 'c'])
    # label_0_incorrect = set([labels_0[i] for i in range(len(labels_0)) if predictions_label_0[i] == 'i'])
    # with open(os.path.join(args.output_dir, 'labels_other_correct'), 'w') as f:
    #     f.write('\n'.join(list(label_0_correct)))
    # with open(os.path.join(args.output_dir, 'labels_other_incorrect'), 'w') as f:
    #     f.write('\n'.join(list(label_0_incorrect)))


        # 3. get original DNA sequence form sequence of kmers
        # 4. show parts of the DNA sequence with the meaningful kmers
        # 7. amongst the species investigated, which ones are part and important to the marine microbiomes
        # 8. map sequences of interest (and less interesting) to the training and testing genome
        # + show the regions of interest (and less interesting) on the genome
        # 9. are the sequences with less relevant kmers less well classified?
        # 10. what can be done with this information to improve taxonomic classification

        # this study can help develop new methods to improve taxonomic classification of metagenomics data



if __name__ == "__main__":
    main()
