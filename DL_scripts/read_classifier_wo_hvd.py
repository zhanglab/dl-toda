import datetime
import tensorflow as tf
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.tfrecord as tfrec
import nvidia.dali.plugin.tf as dali_tf
from AlexNet import AlexNet
from lstm import LSTM
from VDCNN import VDCNN
from VGG16 import VGG16
from DNA_model_1 import DNA_net_1
from DNA_model_2 import DNA_net_2
from transformers import TFBertForSequenceClassification, BertConfig
import os
import sys
import json
import glob
import time
import numpy as np
import math
import argparse
import random


# set seed
seed = 42
# set seed for tensorflow
tf.random.set_seed(seed)
# set seed for numpy operations
np.random.seed(seed)
# set the global python random seed
random.seed(seed)
# activate tensorflow deterministic behavior
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# set the number of threads used for parallel execution of independent operations to 1
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

dl_toda_dir = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])

# disable eager execution
# tf.compat.v1.disable_eager_execution()
print(f'Is eager execution enabled: {tf.executing_eagerly()}')

# print which unit (CPU/GPU) is used for an operation
#tf.debugging.set_log_device_placement(True)

# enable XLA = XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that can accelerate
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# define the DALI pipeline for CNN
@pipeline_def
def dali_pipeline(tfrec_filenames, tfrec_idx_filenames, shard_id, initial_fill, num_gpus, training=True):
    stick_to_shard = True
    inputs = fn.readers.tfrecord(path=tfrec_filenames,
                                 index_path=tfrec_idx_filenames,
                                 random_shuffle=training,
                                 shard_id=shard_id,
                                 num_shards=num_gpus,
                                 initial_fill=initial_fill,
                                 stick_to_shard=stick_to_shard,
                                 features={
                                     "read": tfrec.VarLenFeature([], tfrec.int64, 0),
                                     "label": tfrec.FixedLenFeature([1], tfrec.int64, -1)})
    # retrieve reads and labels and copy them to the gpus
    reads = inputs["read"].gpu()
    labels = inputs["label"].gpu()
    return reads, labels

# define the DALI pipeline for BERT 
@pipeline_def
def bert_dali_pipeline(tfrec_filenames, tfrec_idx_filenames, shard_id, initial_fill, num_gpus, training=True):
    stick_to_shard = True
    inputs = fn.readers.tfrecord(path=tfrec_filenames,
                                 index_path=tfrec_idx_filenames,
                                 random_shuffle=training,
                                 shard_id=shard_id,
                                 num_shards=num_gpus,
                                 stick_to_shard=stick_to_shard,
                                 initial_fill=initial_fill,
                                 features={
                                     "input_ids": tfrec.VarLenFeature([], tfrec.int64, 0),
                                     "attention_mask": tfrec.VarLenFeature([], tfrec.int64, 0),
                                     "position_ids": tfrec.VarLenFeature([], tfrec.int64, 0),
                                     "token_type_ids": tfrec.VarLenFeature([], tfrec.int64, 0),
                                     "labels": tfrec.FixedLenFeature([1], tfrec.int64, -1)})
    
    # retrieve data and copy it to the gpus
    input_ids = inputs["input_ids"].gpu()
    attention_mask = inputs["attention_mask"].gpu()
    token_type_ids = inputs["token_type_ids"].gpu()
    position_ids = inputs["position_ids"].gpu()
    labels = inputs["labels"].gpu()

    return (input_ids, attention_mask, position_ids, token_type_ids, labels)


class DALIPreprocessor(object):
    def __init__(self, model_type, filenames, idx_filenames, batch_size, vector_size, initial_fill, deterministic=False, training=False):

        device_id = 0
        shard_id = 0
        num_gpus = 1
        
        # self.batch_size = batch_size
        # self.device_id = device_id

        if model_type == "BERT":
            self.pipe = bert_dali_pipeline(tfrec_filenames=filenames, tfrec_idx_filenames=idx_filenames, batch_size=batch_size,
                                      device_id=device_id, shard_id=shard_id, initial_fill=initial_fill, num_gpus=num_gpus,
                                      training=training, seed=7 if deterministic else None)

            self.dalidataset = dali_tf.DALIDataset(fail_on_device_mismatch=False, pipeline=self.pipe,
                output_shapes=((batch_size, vector_size), (batch_size, vector_size), (batch_size, vector_size), (batch_size, vector_size), (batch_size)),
                batch_size=batch_size, output_dtypes=(tf.int64, tf.int64, tf.int64, tf.int64, tf.int64), device_id=device_id)
        else:
            self.pipe = dali_pipeline(tfrec_filenames=filenames, tfrec_idx_filenames=idx_filenames, batch_size=batch_size,
                                      device_id=device_id, shard_id=shard_id, initial_fill=initial_fill, num_gpus=num_gpus,
                                      training=training, seed=7 if deterministic else None)
   
            self.dalidataset = dali_tf.DALIDataset(fail_on_device_mismatch=False, pipeline=self.pipe,
                output_shapes=((batch_size, vector_size), (batch_size)),
                batch_size=batch_size, output_dtypes=(tf.int64, tf.int64), device_id=device_id)

    def get_device_dataset(self):
        return self.dalidataset


def build_dataset_sim(args, filenames, num_classes, is_training, drop_remainder):

    def load_tfrecords_with_reads(proto_example):
        data_description = {
            'read': tf.io.VarLenFeature(tf.int64),
            'label': tf.io.FixedLenFeature([1], tf.int64)
        }
        # load one example
        parsed_example = tf.io.parse_single_example(serialized=proto_example, features=data_description)
        read = parsed_example['read']
        label = tf.cast(parsed_example['label'], tf.int64)
        read = tf.sparse.to_dense(read)
        return read, label

    def load_tfrecords_for_finetuning(proto_example):
        name_to_features = {
          "input_ids": tf.io.FixedLenFeature([args.vector_size], tf.int64),
          "attention_mask": tf.io.FixedLenFeature([args.vector_size], tf.int64),
          "position_ids": tf.io.FixedLenFeature([args.vector_size], tf.int64),
          "token_type_ids": tf.io.FixedLenFeature([args.vector_size], tf.int64),
          "labels": tf.io.FixedLenFeature([1], tf.int64)
        }
        parsed_example = tf.io.parse_single_example(serialized=proto_example, features=name_to_features)

        return {"input_ids": parsed_example['input_ids'], "position_ids": parsed_example['position_ids'], "token_type_ids": parsed_example['token_type_ids'], "attention_mask": parsed_example['attention_mask'], "labels": parsed_example['labels']}

    """ Return data in TFRecords """
    fn_load_data = {'reads': load_tfrecords_with_reads, 'finetuning': load_tfrecords_for_finetuning}

    dataset = tf.data.TFRecordDataset([filenames])

    if is_training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.map(map_func=fn_load_data[args.datatype])
    dataset = dataset.batch(args.batch_size, drop_remainder=drop_remainder)


    # Load data as shards
    # dataset = tf.data.Dataset.list_files(tfrecord_path)
    # dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x), num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                 # deterministic=False)
    # dataset = dataset.map(map_func=fn_load_data[datatype], num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.padded_batch(batch_size,
                                   # padded_shapes=(tf.TensorShape([vector_size]), tf.TensorShape([num_classes])),)
    # dataset = dataset.cache()
    # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


# @tf.function
# def testing_step(data_type, reads, labels, model, loss=None, test_loss=None, test_accuracy=None, target_label=None):
#     print('inside testing_step')
#     probs = model(reads, training=False)
#     if data_type == 'test':
#         test_accuracy.update_state(labels, probs)
#         loss_value = loss(labels, probs)
#         test_loss.update_state(loss_value)
#     pred_labels = tf.math.argmax(probs, axis=1)
#     pred_probs = tf.reduce_max(probs, axis=1)
#     if target_label:
#         label_prob = tf.gather(probs, target_label, axis=1)
#     return probs, pred_labels, pred_probs
#     # return pred_labels, pred_probs, label_prob

@tf.function
def testing_step(model_type, bert_step, data, model, loss=None, test_loss=None, test_accuracy=None, target_label=None, nvidia_dali=False):
    training = False

    if model_type == 'BERT':
        if nvidia_dali:
            input_ids, attention_mask, position_ids, token_type_ids, labels = data
        else:
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            token_type_ids = data["token_type_ids"]
            position_ids = data["position_ids"]
            labels = data["labels"]

    if bert_step in ['finetuning', 'regular']:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        # outputs = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, labels=labels)
        # outputs = model(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
        # outputs = model(**data)
        # logits = model(**data).logits
        # loss_value = model(**data).loss
        # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        # probs = tf.nn.softmax(logits, axis=-1)
        # labels = data["labels"]
        logits = outputs.logits
        probs = tf.nn.softmax(logits, axis=-1)
        loss_value = loss(labels, probs)
    else:
        reads, labels = data
        probs = model(reads, training=training)
    
    # if data_type == 'sim':
    test_accuracy.update_state(labels, probs)
    loss_value = loss(labels, probs)
    test_loss.update_state(loss_value)

    # get predicted labels and confidence scores
    pred_labels = tf.math.argmax(probs, axis=1)
    if tf.shape(probs)[1] == 2:
        pred_probs = probs
    else:
        pred_probs = tf.reduce_max(probs, axis=1)

    # if target_label:
    #     label_prob = tf.gather(probs, target_label, axis=1)

    if model_type == 'BERT':
        return pred_labels, pred_probs, labels, outputs
    else:
        return pred_labels, pred_probs, labels


def main():
    start = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords', type=str, help='path to tfrecords', required=True)
    parser.add_argument('--output_dir', type=str, help='directory to store results', default=os.getcwd())
    parser.add_argument('--init_lr', type=float, help='initial learning rate', default=0.0001)
    parser.add_argument('--bert_step', choices=['pretraining', 'finetuning', 'regular'], required=('BERT' in sys.argv))
    parser.add_argument('--batch_size', type=int, help='batch size per gpu', default=8192)
    parser.add_argument('--DNA_model', action='store_true', default=False)
    parser.add_argument('--n_rows', type=int, default=50)
    parser.add_argument('--n_cols', type=int, default=5)
    parser.add_argument('--num_labels', type=int, help='number of labels', default=2)
    parser.add_argument('--nvidia_dali', action='store_true', default=False, required=('val_idx_files' in sys.argv and 'train_idx_files' in sys.argv))
    parser.add_argument('--k_value', type=int, help='length of kmer strings', default=12)
    parser.add_argument('--target_label', type=int, help='output prediction scores of target label')
    parser.add_argument('--embedding_size', type=int, help='size of embedding vectors', default=60)
    parser.add_argument('--dropout_rate', type=float, help='dropout rate to apply to layers', default=0.7)
    parser.add_argument('--vector_size', type=int, help='size of input vectors')
    parser.add_argument('--vocab', help="Path to the vocabulary file", required=('AlexNet' in sys.argv))
    parser.add_argument('--model_type', type=str, help='type of model', choices=['DNA_1', 'DNA_2', 'AlexNet', 'VGG16', 'VDCNN', 'LSTM', 'BERT'])
    parser.add_argument('--bert_config_file', type=str, help='path to bert config file', required=('BERT' in sys.argv))
    parser.add_argument('--model', type=str, help='path to directory containing model in SavedModel format or in the new keras format (provide filename as well)')
    parser.add_argument('--class_mapping', type=str, help='path to json file containing dictionary mapping taxa to labels', default=os.path.join(dl_toda_dir, 'data', 'species_labels.json'))
    parser.add_argument('--ckpt', type=str, help='path to checkpoint file (only add the prefix)')
    parser.add_argument('--pretrained', type=str, help='path to directory containing hf pretrained model saved using save_pretrained')
    parser.add_argument('--max_read_size', type=int, help='maximum read size in training dataset', default=250)
    parser.add_argument('--initial_fill', type=int, help='size of the buffer for random shuffling', default=10000)
    parser.add_argument('--sequences_file', type=str, help='path to tsv or fna file', required=True)
    # parser.add_argument('--save_probs', help='save probability distributions', action='store_true')
    args = parser.parse_args()

    # process one sequence at a time to get embeddings
    args.batch_size = 1

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        # tf.config.experimental.set_visible_devices(gpus, 'GPU')
        tf.config.experimental.set_visible_devices(gpus, 'GPU')

    models = {'DNA_1': DNA_net_1, 'DNA_2': DNA_net_2, 'AlexNet': AlexNet, 'VGG16': VGG16, 'VDCNN': VDCNN, 'LSTM': LSTM}

    # get vocabulary
    kmers = []
    vocab = {}
    # if args.model_type != 'BERT':
    with open(f'{args.vocab}/{args.k_value}mers.txt', 'r') as f:
        for idx, line in enumerate(f,0):
            vocab[idx] = line.rstrip()
            if line.rstrip() not in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']:
                kmers.append(line.rstrip())
        vocab_size = len(vocab)

    print(kmers)
    print(vocab)
    print(len(kmers))

    # load class_mapping file mapping label IDs to species
    if args.class_mapping:
        f = open(args.class_mapping)
        class_mapping = json.load(f)
        num_labels = len(class_mapping)
    else:
        num_labels = args.num_labels

    # create dtype policy
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)
    
    # create output directory
    if not os.path.isdir(args.output_dir):
        os.makedirs(os.path.join(args.output_dir))

    if args.model_type == 'BERT':
        with open(args.bert_config_file, "r") as f:
                args.config_dict = json.load(f)
        args.vector_size = args.config_dict['max_position_embeddings']
        if args.model is not None:
            model = tf.keras.models.load_model(args.model)
        else:
            bert_config = BertConfig(vocab_size=args.config_dict["vocab_size"])
            model = TFBertForSequenceClassification.from_pretrained(args.pretrained, config=bert_config)
    else:
        if args.model is not None:
            model = tf.keras.models.load_model(args.model)
        elif args.ckpt is not None:
            model = models[args.model_type](args, args.vector_size, args.embedding_size, num_labels, vocab_size, args.dropout_rate)
            # define the optimizer
            opt = tf.keras.optimizers.Adam(args.init_lr)
            # prevent numeric underflow when using float16
            opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
            checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
            checkpoint.restore(args.ckpt).expect_partial()

    # define metrics
    loss = tf.losses.SparseCategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # get list of testing tfrecords, number of reads per tfrecords and reads id for metagenomic data
    test_files = sorted(glob.glob(os.path.join(args.tfrecords, '*.tfrec')))
    num_reads_files = sorted(glob.glob(os.path.join(args.tfrecords, '*-read_count')))

    # get id and sequence of reads
    reads_seq = {}
    with open(args.sequences_file, 'r') as f:
        content = f.readlines()
        print(len(content))
        reads_seq = {i+1: content[i].rstrip().split('\t')[1].split(' ') for i in range(len(content))}

    if args.nvidia_dali:
        # get nvidia dali indexes
        test_idx_files = sorted(glob.glob(os.path.join(args.tfrecords, 'idx_files', '*.idx')))
    
    elapsed_time = []
    num_reads_classified = 0
    for i in range(len(test_files)):
        print(test_files[i])
        start_time = time.time()
        # get number of reads in test file
        with open(os.path.join(args.tfrecords, num_reads_files[i]), 'r') as f:
            num_reads = int(f.readline())
        print(f'number of reads to classify: {num_reads}')
        num_reads_classified += num_reads

        # compute number of steps required to iterate over entire test set
        test_steps = math.ceil(num_reads/(args.batch_size))

        # load data
        if args.nvidia_dali:
            nvidia_dali = True
            test_preprocessor = DALIPreprocessor(args.model_type, test_files[i], test_idx_files[i], args.batch_size, args.vector_size, args.initial_fill, deterministic=False, training=False)

            test_input = test_preprocessor.get_device_dataset()
        else:
            nvidia_dali=False
            if args.model_type == 'BERT':
                if args.bert_step in ['finetuning', 'regular']:
                    args.datatype = 'finetuning'
                else:
                    args.datatype = 'pretraining'
                    args.num_masked = int(args.masked_lm_prob * (args.vector_size-1)) # without NSP task
            else:
                args.datatype = 'reads'
            test_input = build_dataset_sim(args, test_files[i], num_labels, is_training=False, drop_remainder=False)

        # create empty arrays to store the predicted and true values, the confidence scores and the probability distributions
        # all_predictions = tf.zeros([args.batch_size, NUM_CLASSES], dtype=tf.dtypes.float32, name=None)
        all_pred_sp = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]
        all_prob_sp = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]
        all_labels = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]
        # all_prob_labels = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]
        embeddings = defaultdict(list) # key = token, value = embeddings
        for batch, data in enumerate(test_input.take(test_steps), 1):
            print(f'batch: {batch}')
            # batch_predictions, batch_pred_sp, batch_prob_sp = testing_step(args.data_type, reads, labels, model, loss, test_loss, test_accuracy)
            # batch_pred_sp, batch_prob_sp, batch_label_prob = testing_step(args.data_type, reads, labels, model, loss, test_loss, test_accuracy, args.target_label)
            # batch_pred_sp, batch_prob_sp, labels = testing_step(args.data_type, args.model_type, args.bert_step, data, model, loss, test_loss, test_accuracy, nvidia_dali=nvidia_dali)
            if args.model_type == 'BERT':
                batch_pred_sp, batch_prob_sp, labels, outputs = testing_step(args.model_type, args.bert_step, data, model, loss, test_loss, test_accuracy, nvidia_dali=nvidia_dali)
                hidden_states = outputs.hidden_states
                print(hidden_states[0].shape)   # shape : (batch_size, sequence_length, hidden_size)
                token_embeddings = hidden_states[0]
                print(token_embeddings)
                seq_ids = data["input_ids"].numpy()[0]
                tokens = [vocab[i] for i in seq_ids]
                assert '[UKN]' not in tokens
                # reconstruct original sequence
                dna_seq = tokens[1]
                for j in range(2, len(tokens), 1):
                    if tokens[j] not in ['[PAD]', '[SEP]', '[UNK]']:
                        dna_seq += tokens[j][-1]
                original_dna_seq = reads_seq[batch][0]
                for j in range(1, len(reads_seq[batch]), 1):
                    original_dna_seq += reads_seq[batch][j][-1]
                assert dna_seq == original_dna_seq, f'{len(dna_seq)}\t{len(tokens)}\t{len(seq_ids)}\n{dna_seq}\n{reads_seq[batch]}\n{tokens}\n{seq_ids}'
                # get embeddings for each token
                for j in range(len(tokens)):
                    embeddings[tokens[j]].append(token_embeddings[0][j])
                print(embeddings)
                sys.exit(1)
            else:
                batch_pred_sp, batch_prob_sp, labels = testing_step(args.model_type, args.bert_step, data, model, loss, test_loss, test_accuracy, nvidia_dali=nvidia_dali)
            if batch == 1:
                all_labels = [labels]
                all_pred_sp = [batch_pred_sp]
                all_prob_sp = [batch_prob_sp]
                # all_prob_labels = [batch_label_prob]
                # all_predictions = batch_predictions
            else:
                # all_predictions = tf.concat([all_predictions, batch_predictions], 0)
                all_pred_sp = tf.concat([all_pred_sp, [batch_pred_sp]], 1)
                all_prob_sp = tf.concat([all_prob_sp, [batch_prob_sp]], 1)
                all_labels = tf.concat([all_labels, [labels]], 1)
                # all_prob_labels = tf.concat([all_prob_labels, [batch_label_prob]], 1)

        # get list of true species, predicted species and predicted probabilities
        # all_predictions = all_predictions.numpy()
        all_pred_sp = all_pred_sp[0].numpy()
        all_prob_sp = all_prob_sp[0].numpy()
        all_labels = all_labels[0].numpy()
        # all_prob_labels = all_prob_labels[0].numpy()
        print(f'before adjusting: {len(all_pred_sp)}\t{len(all_prob_sp)}\t{len(all_labels)}\n')


        # adjust the list of predicted species and read ids if necessary
        if len(all_labels) > num_reads:
            num_extra_reads = (test_steps*args.batch_size) - num_reads
            # all_predictions = all_predictions[:-num_extra_reads]
            all_pred_sp = all_pred_sp[:-num_extra_reads]
            all_prob_sp = all_prob_sp[:-num_extra_reads]
            all_labels = all_labels[:-num_extra_reads]
            print(f'number of reads: {num_extra_reads}\t{num_reads}\t{len(all_pred_sp)}\t{len(all_prob_sp)}\t{len(all_labels)}\n')
            print(all_pred_sp[0], all_prob_sp[0], all_labels[0])
            # all_prob_labels = all_prob_labels[:-num_extra_reads]

        # write results to file
        out_filename = os.path.join(args.output_dir, 'testing-results.tsv')
        # out_filename = os.path.join(args.output_dir, f'{test_files[i].split("/")[-1].split(".")[0]}-out.tsv') if len(test_files[i].split("/")[-1].split(".")) == 2 else os.path.join(args.output_dir, f'{".".join(test_files[i].split("/")[-1].split(".")[0:2])}-out.tsv')
        with open(out_filename, 'w') as out_f:
            for j in range(num_reads):
                if nvidia_dali:
                    out_f.write(f'{all_labels[j]}\t{all_pred_sp[j]}\t{all_prob_sp[j][all_pred_sp[j]]}\n')
                else:
                    out_f.write(f'{all_labels[j][0]}\t{all_pred_sp[j]}\t{all_prob_sp[j][all_pred_sp[j]]}\n')
                # out_f.write(f'{all_labels[j]}\t{all_pred_sp[j]}\t{all_prob_sp[j]}\t{all_prob_labels[j]}\n')
                # if len(all_prob_sp[j]) == num_labels:
                    # out_f.write(f'{all_prob_sp[j][0]}\t{all_prob_sp[j][1]}\n')
                # else:
                # out_f.write(f'{all_prob_sp[j][all_pred_sp[j]]}\n')
        # if args.save_probs:
        #     # save predictions and labels to file
        #     np.save(os.path.join(args.output_dir, f'{gpu_test_files[i].split("/")[-1].split(".")[0]}-prob-out.npy'), all_predictions)
        #     np.save(os.path.join(args.output_dir, f'{gpu_test_files[i].split("/")[-1].split(".")[0]}-labels-out.npy'), all_labels)
        end_time = time.time()
        # elapsed_time = np.append(elapsed_time, end_time - start_time)
        elapsed_time.append(end_time - start_time)

    end = datetime.datetime.now()
    total_time = end - start
    hours, seconds = divmod(total_time.seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    with open(os.path.join(args.output_dir, f'testing-summary.tsv'), 'w') as outfile:
        outfile.write(f'{args.batch_size}\t{len(test_files)}\t{num_reads_classified}\t')
        outfile.write(f'{test_accuracy.result().numpy()}\t{test_loss.result().numpy()}\t')
        if args.ckpt:
            outfile.write(f'{args.ckpt}')
        outfile.write(f'\t{hours}:{minutes}:{seconds}:{total_time.microseconds}\t')

        if len(elapsed_time) > 1:
            outfile.write(f'{(num_reads_classified / sum(elapsed_time))} reads/sec\n')
        else:
            outfile.write(f'{(num_reads_classified / elapsed_time[0])} reads/sec\n')


if __name__ == "__main__":
    main()
