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
    parser.add_argument('--vocab', help="Path to the vocabulary file", required=('AlexNet' in sys.argv))
    parser.add_argument('--bert_config_file', type=str, help='path to bert config file', required=('BERT' in sys.argv or 'BERT_HUGGINGFACE' in sys.argv))
    parser.add_argument('--class_mapping', type=str, help='path to json file containing dictionary mapping taxa to labels', default=os.path.join(dl_toda_dir, 'data', 'species_labels.json'))
    parser.add_argument('--ckpt', type=str, help='path to directory containing checkpoint file')
    parser.add_argument('--model', type=str, help='path to model saved with model.save()')
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
    
    # create BERT config object + model
    bert_config = BertConfig(vocab_size=args.config_dict["vocab_size"])
    # make output of attentions possible
    bert_config.output_attentions=True
    print(bert_config)
    
    # load weights from checkpoint file created with tf.train.Checkpoint() and checkpoint.save()
    model = TFBertForSequenceClassification(config=bert_config)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
    checkpoint.restore(os.path.join(args.ckpt, f'ckpt-best-1')).expect_partial()

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

    # get labels from class 0 
    with open(args.tsv_file, 'r') as f:
        content = f.readlines()
        all_labels = [line.rstrip().split('\t')[0] for line in content]

    args.datatype = 'finetuning'
    test_input = build_dataset(args, test_file, num_labels, is_training=False, drop_remainder=False)

    attention_weights_label_0 = []
    # kmers_label_0 = []
    predictions_label_0 = []
    confidence_scores_label_0 = []
    # labels_0 = []

    attention_weights_label_1 = []
    # kmers_label_1 = []
    predictions_label_1 = []
    confidence_scores_label_1 = []

    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # set color palette
    palette = sn.color_palette("icefire", as_cmap=True)

    print(len(all_labels), test_steps)

    for batch, data in enumerate(test_input.take(test_steps), 1):
        outputs, pred_labels, pred_probs = get_attentions(data, model, test_accuracy)
        # get attentions weights from the 12 attention heads in each of the 12 attention layers
        attentions = list(outputs[-1])
        # print number of attention layers
        # print(len(attentions))
        # print dimensions of the output of the last attention layer
        # print(attentions[-1].shape)
        # shape of the attentions output: (batch_size, num_attention_head, max_position_embeddings, max_position_embeddings)
        # shape of the last attention head output: (max_position_embeddings, max_position_embeddings)

        # print(f'accuracy: {test_accuracy.result().numpy()}')

        for i in range(len(data["input_ids"])):
            label = data["labels"][i].numpy()
            seq_ids = data["input_ids"][i].numpy()
            seq_kmers = [vocab[i] for i in seq_ids]
            # get attention weights of the last attention head in the last attention layer for the sequence investigated, shape is (max_position_embeddings, max_position_embeddings)
            attentions_weights = attentions[-1][-1][i].numpy()
            df = pd.DataFrame(attentions_weights)
            df.columns = seq_kmers
            # remove columns and rows [PAD]
            pad_idx = [i for i in range(len(seq_kmers)) if seq_kmers[i] == '[PAD]']
            df = df.drop(pad_idx, axis='index')
            # remove columns ['PAD']
            df = df.drop('[PAD]', axis='columns')
            # get kmers with high attention weights
            filtered_df = df.loc[:, (df >= np.mean(df.values.tolist())).any()]
            # plot heatmap of attention weights
            plt.figure(figsize=(15, 15))
            sn.heatmap(data=df, annot=False, xticklabels=df.columns, yticklabels=df.columns, cmap=palette) 
            if pred_labels[i] == label:
                plt.savefig(os.path.join(args.output_dir, f'attention_weights_correct_heatmap_{batch}_{len(df)}_{all_labels[batch]}.png'))
            else:
                plt.savefig(os.path.join(args.output_dir, f'attention_weights_incorrect_heatmap_{batch}_{len(df)}_{all_labels[batch]}.png'))

            if label == 0:
                attention_weights_label_0.append(df.values.flatten().tolist())
                # kmers_label_0 += filtered_df.columns.tolist()
                confidence_scores_label_0.append(pred_probs[i])
                # labels_0.append(label)
                if pred_labels[i] == label:
                    predictions_label_0.append('c') 
                else:
                    predictions_label_0.append('i')
            else:
                attention_weights_label_1.append(df.values.flatten().tolist())
                # kmers_label_1 += filtered_df.columns.tolist()
                confidence_scores_label_1.append(pred_probs[i])
                if pred_labels[i] == label:
                    predictions_label_1.append('c') 
                else:
                    predictions_label_1.append('i') 
    
    

    # plot histogram of attention weights for other labels
    confidence_scores_label_0_correct = [confidence_scores_label_0[i] for i in range(len(confidence_scores_label_0)) if predictions_label_0[i] == 'c']
    confidence_scores_label_0_incorrect = [confidence_scores_label_0[i] for i in range(len(confidence_scores_label_0)) if predictions_label_0[i] == 'i']
    plt.figure(figsize=(10, 6))
    sn.histplot(data=confidence_scores_label_0_correct)
    plt.xlabel('Confidence Scores')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'confidence_scores_correct_hist_other.png'))
    plt.figure(figsize=(10, 6))
    sn.histplot(data=confidence_scores_label_0_incorrect)
    plt.xlabel('Confidence Scores')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'confidence_scores_incorrect_hist_other.png'))

    # plot histogram of attention weights for label investigated
    confidence_scores_label_1_correct = [confidence_scores_label_1[i] for i in range(len(confidence_scores_label_1)) if predictions_label_1[i] == 'c']
    confidence_scores_label_1_incorrect = [confidence_scores_label_1[i] for i in range(len(confidence_scores_label_1)) if predictions_label_1[i] == 'i']
    plt.figure(figsize=(10, 6))
    sn.histplot(data=confidence_scores_label_1_correct)
    plt.xlabel('Confidence Scores')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'confidence_scores_correct_hist_label.png'))
    plt.figure(figsize=(10, 6))
    sn.histplot(data=confidence_scores_label_1_incorrect)
    plt.xlabel('Confidence Scores')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'confidence_scores_incorrect_hist_label.png'))

    # plot histogram of confidence scores for other labels
    attention_weights_label_0_correct = [attention_weights_label_0[i] for i in range(len(attention_weights_label_0)) if predictions_label_0[i] == 'c']
    attention_weights_label_0_incorrect = [attention_weights_label_0[i] for i in range(len(attention_weights_label_0)) if predictions_label_0[i] == 'i']
    plt.figure(figsize=(10, 6))
    sn.histplot(data=attention_weights_label_0_correct)
    plt.xlabel('Attention Weights')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'attention_weights_correct_hist_other.png'))
    plt.figure(figsize=(10, 6))
    sn.histplot(data=attention_weights_label_0_incorrect)
    plt.xlabel('Attention Weights')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'attention_weights_incorrect_hist_other.png'))

    # plot histogram of attention weights for label investigated
    attention_weights_label_1_correct = [attention_weights_label_1[i] for i in range(len(attention_weights_label_1)) if predictions_label_1[i] == 'c']
    attention_weights_label_1_incorrect = [attention_weights_label_1[i] for i in range(len(attention_weights_label_1)) if predictions_label_1[i] == 'i']
    plt.figure(figsize=(10, 6))
    sn.histplot(data=attention_weights_label_1_correct)
    plt.xlabel('Attention Weights')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'attention_weights_correct_hist_label.png'))
    plt.figure(figsize=(10, 6))
    sn.histplot(data=attention_weights_label_1_incorrect)
    plt.xlabel('Attention Weights')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'attention_weights_incorrect_hist_label.png'))

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
