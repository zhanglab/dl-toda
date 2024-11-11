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
def get_attentions(data, model):
    outputs = model(input_ids=data["input_ids"], attention_mask=data["attention_mask"], token_type_ids=data["token_type_ids"])
    # outputs = model(data)
    # outputs = model(**data)
    attentions = outputs[-1]

    return attentions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords', type=str, help='path to tfrecords', required=True)
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
    
    # model = TFBertForSequenceClassification(config=bert_config)
    # checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
    # checkpoint.restore(os.path.join(args.ckpt, f'ckpt-best-1')).expect_partial()
    
    model = TFBertForSequenceClassification(config=bert_config)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(args.model).expect_partial()

    # update input vector size
    args.vector_size = args.config_dict['max_position_embeddings']
    
    # checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
    # following command fails, not all variables in the checkpoint file have been loaded into the current model
    # checkpoint.restore(os.path.join(args.ckpt, f'ckpt-best-1')).assert_consumed()
    
    # use .expect_partial() to restore only a subset of the variables
    # checkpoint.restore(os.path.join(args.ckpt, f'ckpt-best-1')).expect_partial()
    # get weights from model check that the weights are always the same after loading checkpoint
    # all_weights = model.get_weights() 
    # print(all_weights)
    # --> ValueError: Weights for model 'tf_bert_for_sequence_classification' have not yet been created. Weights are created when the model is first called on inputs or `build()` is called with an `input_shape`.



    # to be used with save_weights
    # model.load_weights(os.path.join(args.ckpt, f'ckpt-best-1'))
    # all_weights = model.get_weights() 
    # print(all_weights)

    # model = tf.keras.models.load_model(args.model, compile=False)

    # all_weights = model.get_weights() 
    # print(all_weights)
    # --> ValueError: Could not find matching concrete function to call loaded from the SavedModel.




    # get name of layers in model
    # layer_names = [layer.name for layer in model.layers]
    # print(layer_names)
    # ['bert', 'dropout_37', 'classifier']


    # get list of testing tfrecords and number of reads per tfrecords
    test_file = sorted(glob.glob(os.path.join(args.tfrecords, '*.tfrec')))
    num_reads_file = sorted(glob.glob(os.path.join(args.tfrecords, '*-read_count')))

    with open(num_reads_file[0], 'r') as infile:
        num_reads = int(infile.readline())
    print(f'# sequences: {num_reads}')
    # compute number of steps required to iterate over entire test set
    test_steps = math.ceil(num_reads/(args.batch_size))

    args.datatype = 'finetuning'
    test_input = build_dataset(args, test_file, num_labels, is_training=False, drop_remainder=False)

    for batch, data in enumerate(test_input.take(test_steps), 1):
        attentions = get_attentions(data, model)
        print(f'attentions : {len(attentions)}') 
        # shape of the attentions output: (batch_size, num_attention_head, max_position_embeddings, max_position_embeddings)
        print(attentions[-1].shape)
        # shape of the last attention head output: (max_position_embeddings, max_position_embeddings)
        print(attentions[-1][-1].shape)

        # get kmers of ids
        print(f'input ids: {data["input_ids"]}')
        # (batch_size, max_position_embeddings)
        print(data["input_ids"].shape)
        for i in range(len(data["input_ids"])):
            seq_ids = data["input_ids"][i].numpy()
            seq_kmers = [vocab[i] for i in seq_ids]
            # print(seq_ids)
            # print(seq_kmers)
            # get attention weights of the last attention head for the sequence investigated, shape is (max_position_embeddings, max_position_embeddings)
            # print(attentions[-1][-1][i].shape)
            attentions_weights = attentions[-1][-1][i].numpy()
            print(attentions_weights)
            # plot heatmap of attention weights
            df = pd.DataFrame(attentions_weights)
            df.columns = seq_kmers
            # remove columns and rows [PAD]
            pad_idx = [i for i in range(len(seq_kmers)) if seq_kmers[i] == '[PAD]']
            print(len(pad_idx))
            print(df.shape)
            # remove rows ['PAD']
            df = df.drop(pad_idx, axis='index')
            print(df.shape)
            # remove columns ['PAD']
            df = df.drop('[PAD]', axis='columns')
            print(df.shape)
            print(df)
            # set color palette
            palette = sn.color_palette("icefire", as_cmap=True)
            plt.figure(figsize=(15, 15))
            sn.heatmap(data=df, annot=False, xticklabels=df.columns, yticklabels=df.columns, cmap=palette) 
            plt.savefig(os.path.join(args.output_dir, 'attention_weights_heatmap.png'))
            # plot histogram of attention weights
            plt.figure(figsize=(10, 6))
            sn.histplot(data=df.values.tolist())
            plt.xlabel('Attention Weights')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(os.path.join(args.output_dir, 'attention_weights_hist.png'))
            # get stats on attention weights
            print(f'Stats on attentions:\nMean: {np.mean(df.values.tolist())}\tSd: {np.std(df.values.tolist())}\t'
                f'Median: {np.median(df.values.tolist())}\tMin: {np.min(df.values.tolist())}\tMax: {np.max(df.values.tolist())}\t'
                f'Sum: {np.sum(df.values.tolist())}')
            # get list of relevant kmers

        # 1. get species with high performance
        # 2. find kmers that are attended to each other
        # 3. get original DNA sequence form sequence of kmers
        # 4. show parts of the DNA sequence with the meaningful kmers
        # 5. list the kmers that are relevant
        # 6. get stats on sequences with meaningful kmers
        # 7. amongst the species investigated, which ones are part and important to the marine microbiomes
        # 8. map sequences of interest (and less interesting) to the training and testing genome
        # + show the regions of interest (and less interesting) on the genome
        # 9. are the sequences with less relevant kmers less well classified?
        # 10. what can be done with this information to improve taxonomic classification

        # this study can help develop new methods to improve taxonomic classification of metagenomics data

        break


if __name__ == "__main__":
    main()
