import tensorflow as tf
# import tensorflow_addons as tfa
import tensorflow_models as tfm
import tensorflow_hub as hub
# import horovod.tensorflow as hvd
import tensorflow.keras as keras
from keras import backend as K
from collections import Counter, defaultdict
# from nvidia.dali.pipeline import pipeline_def
# import nvidia.dali.fn as fn
# import nvidia.dali.tfrecord as tfrec
# import nvidia.dali.plugin.tf as dali_tf
from transformers import TFBertForSequenceClassification, BertConfig
import os
import sys
import json
import glob
import datetime
import math
import io
import json
# import numpy as np
from AlexNet import AlexNet
from lstm import LSTM
from VDCNN import VDCNN
from VGG16 import VGG16
from DNA_model_1 import DNA_net_1
from DNA_model_2 import DNA_net_2
from BERT import BertConfiguration, BertModelFinetuning, BertModelPretraining
from optimizers import AdamWeightDecayOptimizer
import argparse

# set seed
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)
tf.experimental.numpy.random.seed(seed)
# activate tensorflow deterministic behavior
#os.environ['TF_DETERMINISTIC_OPS'] = '1'
#os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
#tf.config.threading.set_inter_op_parallelism_threads(1)
#tf.config.threading.set_intra_op_parallelism_threads(1)

dl_toda_dir = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])

# disable eager execution
#tf.compat.v1.disable_eager_execution()
print(f'Is eager execution enabled: {tf.executing_eagerly()}')

# print which unit (CPU/GPU) is used for an operation
#tf.debugging.set_log_device_placement(True)

# enable XLA = XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that can accelerate
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# # define the DALI pipeline
# @pipeline_def
# def get_dali_pipeline(tfrec_filenames, tfrec_idx_filenames, shard_id, initial_fill, num_gpus, training=True):
#     # prefetch_queue_depth = 100
#     # read_ahead = True
#     stick_to_shard = True
#     inputs = fn.readers.tfrecord(path=tfrec_filenames,
#                                  index_path=tfrec_idx_filenames,
#                                  random_shuffle=training,
#                                  shard_id=shard_id,
#                                  num_shards=num_gpus,
#                                  initial_fill=initial_fill,
#                                  # prefetch_queue_depth=prefetch_queue_depth,
#                                  # read_ahead=read_ahead,
#                                  stick_to_shard=stick_to_shard,
#                                  features={
#                                      "read": tfrec.VarLenFeature([], tfrec.int64, 0),
#                                      "label": tfrec.FixedLenFeature([1], tfrec.int64, -1)})
#     # retrieve reads and labels and copy them to the gpus
#     reads = inputs["read"].gpu()
#     labels = inputs["label"].gpu()
#     return (reads, labels)


# # define the BERT DALI pipeline
# @pipeline_def
# def get_bert_dali_pipeline(tfrec_filenames, tfrec_idx_filenames, shard_id, initial_fill, num_gpus, training=True):
#     inputs = fn.readers.tfrecord(path=tfrec_filenames,
#                                  index_path=tfrec_idx_filenames,
#                                  random_shuffle=training,
#                                  shard_id=0,
#                                  num_shards=1,
#                                  stick_to_shard=False,
#                                  initial_fill=initial_fill,
#                                  features={
#                                      "input_ids": tfrec.VarLenFeature([], tfrec.int64, 0),
#                                      "input_mask": tfrec.VarLenFeature([], tfrec.int64, 0),
#                                      "segment_ids": tfrec.VarLenFeature([], tfrec.int64, 0),
#                                      "is_real_example": tfrec.FixedLenFeature([1], tfrec.int64, -1),
#                                      "label_ids": tfrec.FixedLenFeature([1], tfrec.int64, -1)})
#     # retrieve reads and labels and copy them to the gpus
#     input_ids = inputs["input_ids"].gpu()
#     input_mask = inputs["input_mask"].gpu()
#     segment_ids = inputs["segment_ids"].gpu()
#     label_ids = inputs['label_ids'].gpu()
#     is_real_example = inputs['is_real_example'].gpu()

#     return (input_ids, input_mask, segment_ids, label_ids, is_real_example)



# class DALIPreprocessor(object):
#     def __init__(self, args, filenames, idx_filenames, batch_size, vector_size, initial_fill,
#                deterministic=False, training=False):

#         device_id = hvd.local_rank()
#         shard_id = hvd.rank()
#         num_gpus = hvd.size()
        
#         self.batch_size = batch_size
#         self.device_id = device_id

#         if args.model_type == "BERT":
#             self.pipe = get_bert_dali_pipeline(tfrec_filenames=filenames, tfrec_idx_filenames=idx_filenames, batch_size=batch_size,
#                                       device_id=device_id, shard_id=shard_id, initial_fill=initial_fill, num_gpus=num_gpus,
#                                       training=training, seed=7 * (1 + hvd.rank()) if deterministic else None)

#             self.dalidataset = dali_tf.DALIDataset(fail_on_device_mismatch=False, pipeline=self.pipe,
#                 output_shapes=((args.batch_size, vector_size), (args.batch_size, vector_size), (args.batch_size, vector_size), (args.batch_size), (args.batch_size)),
#                 batch_size=batch_size, output_dtypes=(tf.int64, tf.int64, tf.int64, tf.int64, tf.int64), device_id=device_id)
#         else:
#             self.pipe = get_dali_pipeline(tfrec_filenames=filenames, tfrec_idx_filenames=idx_filenames, batch_size=batch_size,
#                                       device_id=device_id, shard_id=shard_id, initial_fill=initial_fill, num_gpus=num_gpus,
#                                       training=training, seed=7 * (1 + hvd.rank()) if deterministic else None)
   
#             self.dalidataset = dali_tf.DALIDataset(fail_on_device_mismatch=False, pipeline=self.pipe,
#                 output_shapes=((batch_size, vector_size), (batch_size)),
#                 batch_size=batch_size, output_dtypes=(tf.int64, tf.int64), device_id=device_id)

#     def get_device_dataset(self):
#         return self.dalidataset


def build_dataset(args, filenames, num_classes, is_training, drop_remainder):

    def load_tfrecords_with_reads(proto_example):
        data_description = {
            'read': tf.io.VarLenFeature(tf.int64),
            # 'read': tf.io.FixedLenFeature([args.vector_size], tf.int64),
            # 'read': tf.io.FixedLenSequenceFeature([args.vector_size], tf.int64, allow_missing=True),
            'label': tf.io.FixedLenFeature([1], tf.int64)
            # 'label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
        }
        parsed_example = tf.io.parse_single_example(serialized=proto_example, features=data_description)
        read = parsed_example['read']
        label = tf.cast(parsed_example['label'], tf.int64)
        read = tf.sparse.to_dense(read)
        # return read, label
        return read


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
    fn_load_data = {'reads': load_tfrecords_with_reads, 'finetuning': load_tfrecords_for_finetuning, 'pretraining': load_tfrecords_for_pretraining}

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

    
# logits = model(**data[0]).logits
#             predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
#             print(predicted_class_id)


@tf.function
def training_step(model_type, bert_step, data, num_labels, train_accuracy, loss, opt, model, first_batch):
    training = True
    with tf.GradientTape() as tape:
        if model_type == 'BERT' and bert_step == "finetuning":
            input_data = (data["input_ids"], data["token_type_ids"], data["attention_mask"])
            labels = data["labels"]
            logits = model(input_data, training=True)
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            probs = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss_value_1 = tf.reduce_mean(per_example_loss)
            loss_value = loss(labels, probs)

        elif model_type == 'BERT' and bert_step == "pretraining":
            input_ids, input_mask, token_type_ids, masked_lm_positions, masked_lm_weights, masked_lm_ids, nsp_label = data
            logits, masked_lm_probs, masked_lm_log_probs, masked_lm_ids, label_ids, masked_lm_weights, label_weights, one_hot_labels, masked_lm_example_loss, numerator, denominator, masked_lm_loss = model(input_ids, input_mask, token_type_ids, masked_lm_positions, masked_lm_weights, masked_lm_ids, nsp_label, training)
            masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
            masked_lm_predictions = tf.argmax(
                masked_lm_log_probs, axis=-1, output_type=tf.int32)
            masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
            masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
            masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
            loss_value_1 = tf.reduce_mean(masked_lm_example_loss)
            loss_value = loss(masked_lm_ids, masked_lm_probs)

        elif model_type == 'BERT_HUGGINGFACE' and bert_step == "finetuning":
            logits = model(**data).logits
            per_example_loss = model(**data).loss
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            probs = tf.nn.softmax(logits, axis=-1)
            labels = data["labels"]
            loss_value = loss(labels, probs)
        else:
            reads, labels = data
            probs = model(reads, training=training)
            # get the loss
            loss_value = loss(labels, probs)
        # scale the loss (multiply the loss by a factor) to avoid numeric underflow
        # scaled_loss = opt.get_scaled_loss(loss_value)
    # use DistributedGradientTape to wrap tf.GradientTape and use an allreduce to
    # combine gradient values before applying gradients to model weights
    # tape = hvd.DistributedGradientTape(tape)
    # get the scaled gradients
    # scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    # get the unscaled gradients
    # grads = opt.get_unscaled_gradients(scaled_gradients)
    grads = tape.gradient(loss_value, model.trainable_variables)
    #opt.apply_gradients(zip(grads, model.trainable_variables))
    opt.apply_gradients(zip(grads, model.trainable_variables))
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    # if first_batch:
    #     print(f'First_batch: {first_batch}')
    #     hvd.broadcast_variables(model.variables, root_rank=0)
    #     hvd.broadcast_variables(opt.variables(), root_rank=0)

    #update training accuracy and loss
    if model_type == 'BERT' and bert_step == "finetuning":
        train_accuracy.update_state(labels, probs)
    elif model_type == 'BERT' and bert_step == "pretraining":
        train_accuracy.update_state(masked_lm_ids, masked_lm_predictions, sample_weight=masked_lm_weights)
    else:
        train_accuracy.update_state(labels, probs)

    return loss_value

@tf.function
def testing_step(model_type, bert_step, data, num_labels, val_accuracy, val_loss, loss, model):
    training = False

    if model_type == 'BERT' and bert_step == "finetuning":
        input_data = (data["input_ids"], data["token_type_ids"], data["attention_mask"])
        labels = data["labels"]
        logits = model(input_data, training=True)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        probs = tf.nn.softmax(logits, axis=-1)
        loss_value = loss(labels, probs)

    elif model_type == 'BERT' and bert_step == "pretraining":
        input_ids, input_mask, token_type_ids, masked_lm_positions, masked_lm_weights, masked_lm_ids, nsp_label = data
        logits, masked_lm_probs, masked_lm_log_probs, masked_lm_ids, label_ids, masked_lm_weights, label_weights, one_hot_labels, masked_lm_example_loss, numerator, denominator, masked_lm_loss = model(input_ids, input_mask, token_type_ids, masked_lm_positions, masked_lm_weights, masked_lm_ids, nsp_label, training)
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(
                masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        loss_value_1 = tf.reduce_mean(masked_lm_example_loss)
        loss_value = loss(masked_lm_ids, masked_lm_probs)

    elif model_type == 'BERT_HUGGINGFACE' and bert_step == "finetuning":
        logits = model(**data).logits
        loss_value = model(**data).loss
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        probs = tf.nn.softmax(logits, axis=-1)
        labels = data["labels"]
        
    else:
        reads, labels = data
        probs = model(reads, training=training)
        loss_value = loss(labels, probs)
    
    if model_type == 'BERT' and bert_step == "finetuning":
        val_accuracy.update_state(labels, probs)
    elif model_type == 'BERT' and bert_step == "pretraining":
        val_accuracy.update_state(masked_lm_ids, masked_lm_predictions, sample_weight=masked_lm_weights)
    else:
        val_accuracy.update_state(labels, probs)
    
    val_loss.update_state(loss_value)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_tfrecords', type=str, help='path to training tfrecords', required=True)
    parser.add_argument('--train_idx_files', type=str, help='path to training dali index files')
    parser.add_argument('--val_tfrecords', type=str, help='path to validation tfrecords', required=True)
    parser.add_argument('--val_idx_files', type=str, help='path to validation dali index files')
    parser.add_argument('--class_mapping', type=str, help='path to json file containing dictionary mapping taxa to labels')
    parser.add_argument('--output_dir', type=str, help='path to store model', default=os.getcwd())
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--bert_step', choices=['pretraining', 'finetuning'], required=('BERT' in sys.argv or 'BERT_HUGGINGFACE' in sys.argv))
    parser.add_argument('--epoch_to_resume', type=int, required=('-resume' in sys.argv))
    parser.add_argument('--num_labels', type=int, help='number of labels', default=2)
    parser.add_argument('--ckpt', type=str, help='full path to checkpoint file with prefix and without .data-00000-of-00001', required=('--resume' in sys.argv))
    parser.add_argument('--model', type=str, help='path to model', required=('-resume' in sys.argv))
    parser.add_argument('--epochs', type=int, help='number of epochs', default=30)
    parser.add_argument('--optimizer', type=str, help='type of optimizer', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--dropout_rate', type=float, help='dropout rate to apply to layers', default=0.7)
    parser.add_argument('--batch_size', type=int, help='batch size per gpu', default=512)
    parser.add_argument('--initial_fill', type=int, help='size of the buffer for random shuffling', default=10000)
    parser.add_argument('--k_value', type=int, help='length of kmer strings', default=12)
    parser.add_argument('--embedding_size', type=int, help='size of embedding vectors', default=60)
    parser.add_argument('--vector_size', type=int, help='size of input vectors')
    parser.add_argument('--vocab', help="Path to the vocabulary file")
    parser.add_argument('--rnd', type=int, help='round of training', default=1)
    parser.add_argument('--model_type', type=str, help='type of model', choices=['DNA_1', 'DNA_2', 'AlexNet', 'VGG16', 'VDCNN', 'LSTM', 'BERT', 'BERT_HUGGINGFACE'])
    parser.add_argument('--bert_config_file', type=str, help='path to bert config file', required=('BERT' in sys.argv or 'BERT_HUGGINGFACE' in sys.argv))
    parser.add_argument('--masked_lm_prob', type=float, help='percentage of token masked', required=('pretraining' in sys.argv), default=0.15)
    parser.add_argument('--path_to_lr_schedule', type=str, help='path to file lr_schedule.py')
    parser.add_argument('--clr', action='store_true', default=False)
    parser.add_argument('--nvidia_dali', action='store_true', default=False, required=('val_idx_files' in sys.argv and 'train_idx_files' in sys.argv))
    parser.add_argument('--DNA_model', action='store_true', default=False)
    parser.add_argument('--paired_reads', action='store_true', default=False)
    parser.add_argument('--with_insert_size', action='store_true', default=False)
    parser.add_argument('--init_lr', type=float, help='initial learning rate', default=0.0001)
    parser.add_argument('--max_lr', type=float, help='maximum learning rate', default=0.001)
    parser.add_argument('--lr_decay', type=int, help='number of epochs before dividing learning rate in half', required=False)
    args = parser.parse_args()

    print(f'VECTOR SIZE: {args.vector_size}')

    # Initialize Horovod
    # hvd.init()
    # Map one GPU per process
    # use hvd.local_rank() for gpu pinning instead of hvd.rank()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # print(f'GPU RANK: {hvd.rank()}/{hvd.local_rank()} - LIST GPUs: {gpus}')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
        # tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    models = {'DNA_1': DNA_net_1, 'DNA_2': DNA_net_2, 'AlexNet': AlexNet, 'VGG16': VGG16, 'VDCNN': VDCNN, 'LSTM': LSTM}

    # get vocabulary size
    if args.model_type not in ['BERT', 'BERT_HUGGINGFACE']:
        with open(f'{args.vocab}/{args.k_value}mers.txt', 'r') as f:
            content = f.readlines()
            vocab_size = len(content)

    # load class_mapping file mapping label IDs to labels and get number of labels
    if args.class_mapping:
        f = open(args.class_mapping)
        class_mapping = json.load(f)
        num_labels = len(class_mapping)
    else:
        num_labels = args.num_labels

    # modify tensorflow precision mode
    # policy = keras.mixed_precision.Policy('mixed_float16')
    # # keras.mixed_precision.set_global_policy(policy)
    # print('Compute dtype: %s' % policy.compute_dtype)
    # print('Variable dtype: %s' % policy.variable_dtype)

    # Get training and validation tfrecords
    train_files = sorted(glob.glob(os.path.join(args.train_tfrecords, '*.tfrec')))
    val_files = sorted(glob.glob(os.path.join(args.val_tfrecords, '*.tfrec')))
    train_num_reads = sorted(glob.glob(os.path.join(args.train_tfrecords, '*-read_count')))
    val_num_reads = sorted(glob.glob(os.path.join(args.val_tfrecords, '*-read_count')))
   
    if args.nvidia_dali:
        # get nvidia dali indexes
        train_idx_files = sorted(glob.glob(os.path.join(args.train_idx_files, 'train*.idx')))
        val_idx_files = sorted(glob.glob(os.path.join(args.val_idx_files, 'val*.idx')))

        # load data
        train_preprocessor = DALIPreprocessor(args, train_files, train_idx_files, args.batch_size, args.vector_size, args.initial_fill,
                                               deterministic=False, training=True)
        val_preprocessor = DALIPreprocessor(args, val_files, val_idx_files, args.batch_size, args.vector_size, args.initial_fill,
                                            deterministic=False, training=True)
        train_input = train_preprocessor.get_device_dataset()
        val_input = val_preprocessor.get_device_dataset()
    else:
        if args.model_type == 'BERT' or args.model_type == 'BERT_HUGGINGFACE':
            if args.bert_step == 'finetuning':
                args.datatype = 'finetuning'
            else:
                args.datatype = 'pretraining'
                args.num_masked = int(args.masked_lm_prob * (args.vector_size-1)) # without NSP task
        else:
            args.datatype = 'reads'

        # drop_remainder set to True prevents smaller batches from being produced (before it what set to True and it worked)
        train_input = build_dataset(args, train_files, num_labels, is_training=True, drop_remainder=False)
        val_input = build_dataset(args, val_files, num_labels, is_training=False, drop_remainder=False)


    # get number of reads in training and validation datasets
    with open(train_num_reads[0], 'r') as infile:
        train_reads_per_epoch = int(infile.readline())

    with open(val_num_reads[0], 'r') as infile:
        val_reads_per_epoch = int(infile.readline())

    # compute number of steps/batches per epoch
    nstep_per_epoch = int(train_reads_per_epoch/args.batch_size)
    # num_train_steps = int(args.train_reads_per_epoch/args.batch_size*args.epochs)
    num_train_steps = math.ceil(train_reads_per_epoch/args.batch_size*args.epochs)
    # num_train_steps = int(args.train_reads_per_epoch/args.batch_size*args.epochs)
    # compute number of steps/batches to iterate over entire validation set
    val_steps = int(val_reads_per_epoch/args.batch_size)
    num_val_steps = int(val_reads_per_epoch/args.batch_size)

    print(f'number of train steps: {num_train_steps}')

    # # compute number of steps/batches per epoch with horovod imported
    # nstep_per_epoch = int(args.train_reads_per_epoch/(args.batch_size*hvd.size()))
    # num_train_steps = int(args.train_reads_per_epoch/(args.batch_size*hvd.size())*args.epochs)
    # # compute number of steps/batches to iterate over entire validation set
    # val_steps = int(args.val_reads_per_epoch/(args.batch_size*hvd.size()))
    # num_val_steps = int(args.val_reads_per_epoch/(args.batch_size*hvd.size()))

    # if hvd.rank() == 0:
    # create output directory
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # create directory for storing checkpoints
    ckpt_dir = os.path.join(args.output_dir, f'ckpts-rnd-{args.rnd}')
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    # create directory for storing logs
    tensorboard_dir = os.path.join(args.output_dir, f'logs-rnd-{args.rnd}')
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    writer = tf.summary.create_file_writer(tensorboard_dir)
    td_writer = open(os.path.join(args.output_dir, f'logs-rnd-{args.rnd}', f'training_data_rnd_{args.rnd}.tsv'), 'w')
    vd_writer = open(os.path.join(args.output_dir, f'logs-rnd-{args.rnd}', f'validation_data_rnd_{args.rnd}.tsv'), 'w')

    # update epoch and learning rate if necessary
    epoch = args.epoch_to_resume + 1 if args.resume else 1
    # init_lr = args.init_lr/(2*(epoch//args.lr_decay)) if args.resume and epoch > args.lr_decay else args.init_lr

    # define cyclical learning rate
    # if args.clr:
    #     init_lr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=args.init_lr,
    #                                               maximal_learning_rate=args.max_lr,
    #                                               scale_fn=lambda x: 1 / (2. ** (x - 1)),
    #                                               step_size=2 * nstep_per_epoch)

    # set up the optimizer
    if args.model_type == 'BERT' or args.model_type == 'BERT_HUGGINGFACE':
        # define linear decay of the learning rate 
        # linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        #     initial_learning_rate=init_lr,
        #     end_learning_rate=0,
        #     decay_steps=num_train_steps)

        # sys.path.append(args.path_to_lr_schedule)
        # from lr_schedule import LinearWarmup

        # # # define linear decay of the learning rate 
        # # # use tf.compat.v1.train.polynomial_decay instead
        # # linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        # # initial_learning_rate=init_lr,
        # # decay_steps=nstep_per_epoch*args.epochs,
        # # end_learning_rate=0.0,
        # # power=1.0,
        # # cycle=False)

        # # # define linear warmup schedule
        # warmup_proportion = 0.1
        # warmup_steps = int(warmup_proportion * num_train_steps)
        # warmup_schedule = tfm.optimization.lr_schedule.LinearWarmup(
        #      warmup_learning_rate = 0,
        #     after_warmup_lr_sched = linear_decay,
        #     warmup_steps = warmup_steps)

        # opt = tf.keras.optimizers.experimental.Adam(learning_rate = warmup_schedule)

        opt = tf.keras.optimizers.Adam(learning_rate=args.init_lr)

    else:
        if args.optimizer == 'Adam':
            opt = tf.keras.optimizers.Adam(args.init_lr)
        elif args.optimizer == 'SGD':
            opt = tf.keras.optimizers.SGD(args.init_lr)

    # prevent numeric underflow when using float16
    # opt = keras.mixed_precision.LossScaleOptimizer(opt)

    # define model
    if args.model_type == 'BERT':
        
        # load BERT configuration
        args.config = BertConfiguration.from_json_file(args.bert_config_file)
        with open(args.bert_config_file, "r") as f:
            args.config_dict = json.load(f)

        if args.bert_step == "finetuning":
            encoder_config = tfm.nlp.encoders.EncoderConfig({
                'type':'bert',
                'bert': args.config_dict
            })
            bert_encoder = tfm.nlp.encoders.build_encoder(encoder_config)
            model = tfm.nlp.models.BertClassifier(network=bert_encoder, num_classes=2)
            # model = BertModelFinetuning(config=args.config)
            # # define a forward pass
            # # input_ids = tf.ones(shape=[args.batch_size, config.seq_length], dtype=tf.int32)
            # # input_mask = tf.ones(shape=[args.batch_size, config.seq_length], dtype=tf.int32)
            # # token_type_ids = tf.ones(shape=[args.batch_size, config.seq_length], dtype=tf.int32)
            # # _ = model(input_ids, input_mask, token_type_ids, False)
            # print(f'summary: {model.create_model().summary()}')
            # tf.keras.utils.plot_model(model.create_model(), to_file=os.path.join(args.output_dir, f'model-bert.png'), show_shapes=True)
            
            # # print(model.summary())
            # with open(os.path.join(args.output_dir, f'model-bert.txt'), 'w+') as f:
            #     model.create_model().summary(print_fn=lambda x: f.write(x + '\n'))
            # print(f'number of parameters: {model.create_model().count_params()}')
            # trainable_params = sum(K.count_params(layer) for layer in model.trainable_weights)
            # non_trainable_params = sum(K.count_params(layer) for layer in model.non_trainable_weights)
            # print(f'# trainable parameters: {trainable_params}')
            # print(f'# non trainable parameters: {non_trainable_params}')
            # print(f'# variables: {len(model.trainable_weights)}')
            # total_params = 0
            # with open(os.path.join(args.output_dir, f'model_trainable_variables_finetuning.txt'), 'w') as f:
            #     for var in model.trainable_weights:
            #         count = 1
            #         for dim in var.shape:
            #             count *= dim
            #         total_params += count
            #         f.write(f'name = {var.name}, shape = {var.shape}\tcount = {count}\ttotal params = {total_params}\n')
            #         print(f'name = {var.name}, shape = {var.shape}\t {count}')
            #     f.write(f'Total params: {total_params}')
            #     print(f'Total params: {total_params}')

            # # print(model.trainable_weights)
            # # print(len(model.trainable_weights))
        elif args.bert_step == "pretraining":
            encoder_config = tfm.nlp.encoders.EncoderConfig({
                'type':'bert',
                'bert': args.config_dict
            })
            bert_encoder = tfm.nlp.encoders.build_encoder(encoder_config)
            model = tfm.nlp.models.BertPretrainer(network=bert_encoder, num_classes=2)
            # model = BertModelPretraining(config=args.config)
            # print(f'summary: {model.create_model().summary()}')
            # tf.keras.utils.plot_model(model.create_model(), to_file=os.path.join(args.output_dir, f'model-bert.png'), show_shapes=True)
            
            # # print(model.summary())
            # with open(os.path.join(args.output_dir, f'model-bert.txt'), 'w+') as f:
            #     model.create_model().summary(print_fn=lambda x: f.write(x + '\n'))
            # print(f'number of parameters: {model.create_model().count_params()}')
            # trainable_params = sum(K.count_params(layer) for layer in model.trainable_weights)
            # non_trainable_params = sum(K.count_params(layer) for layer in model.non_trainable_weights)
            # print(f'# trainable parameters: {trainable_params}')
            # print(f'# non trainable parameters: {non_trainable_params}')
            # print(f'# variables: {len(model.trainable_weights)}')
            # total_params = 0
            # with open(os.path.join(args.output_dir, f'model_trainable_variables_pretraining.txt'), 'w') as f:
            #     for var in model.trainable_weights:
            #         count = 1
            #         for dim in var.shape:
            #             count *= dim
            #         total_params += count
            #         f.write(f'name = {var.name}, shape = {var.shape}\tcount = {count}\ttotal params = {total_params}\n')
            #         print(f'name = {var.name}, shape = {var.shape}\t {count}')
            #     f.write(f'Total params: {total_params}')
            #     print(f'Total params: {total_params}')

    elif args.model_type == 'BERT_HUGGINGFACE':
        with open(args.bert_config_file, "r") as f:
            args.config_dict = json.load(f)
        
        # create BERT config object + model
        bert_config = BertConfig(vocab_size=args.config_dict["vocab_size"])
        model = TFBertForSequenceClassification(config=bert_config)
    
    else:
        model = models[args.model_type](args, args.vector_size, args.embedding_size, num_labels, vocab_size, args.dropout_rate)

    if args.resume:
        if args.model_type == 'BERT' and args.bert_step == "finetuning":
            checkpoint = tf.train.Checkpoint(encoder=model)
            checkpoint.read(os.path.join(args.ckpt, f'ckpt-{args.epoch_to_resume}')).assert_consumed()
        else:
            # load model in SavedModel format
            #model = tf.keras.models.load_model(args.model)
            # load model saved with checkpoints
            checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
            checkpoint.restore(args.ckpt).expect_partial()
            # checkpoint.restore(os.path.join(args.ckpt, f'ckpt-{args.epoch_to_resume}')).expect_partial()


    # if hvd.rank() == 0:
    # create checkpoint object to save model
    checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
        
    # define metrics
    loss = tf.losses.SparseCategoricalCrossentropy()
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    start = datetime.datetime.now()

    # all_labels = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]

    for batch, data in enumerate(train_input.take(num_train_steps), 1):        
        loss_value = training_step(args.model_type, args.bert_step, data, num_labels, train_accuracy, loss, opt, model, batch == 1)
        # if batch == 1:
        #     all_labels = [labels]
        # else:
        #     all_labels = tf.concat([all_labels, [labels]], 1)
        # if batch % 100 == 0 and hvd.rank() == 0:
        if batch % 100 == 0:
            print(f'Epoch: {epoch} - Step: {batch} - learning rate: {opt.learning_rate.numpy()} - Training loss: {loss_value} - Training accuracy: {train_accuracy.result().numpy()*100}')
        # if batch % 1 == 0 and hvd.rank() == 0:
        if batch % 1 == 0:
            # write metrics
            with writer.as_default():
                tf.summary.scalar("learning_rate", opt.learning_rate, step=batch)
                tf.summary.scalar("train_loss", loss_value, step=batch)
                tf.summary.scalar("train_accuracy", train_accuracy.result().numpy(), step=batch)
                writer.flush()
            td_writer.write(f'{epoch}\t{batch}\t{opt.learning_rate.numpy()}\t{loss_value}\t{train_accuracy.result().numpy()}\n')

        # evaluate model at the end of every epoch
        if batch % nstep_per_epoch == 0:
            # evaluate model
            for batch, data in enumerate(val_input.take(val_steps)):
                # testing_step(args.model_type, args.bert_step, data, num_labels, val_accuracy, val_loss, loss, model)

            # adjust learning rate
            if args.lr_decay:
                if epoch % args.lr_decay == 0:
                    current_lr = opt.learning_rate
                    new_lr = current_lr / 2
                    opt.learning_rate = new_lr

            # if hvd.rank() == 0:
            print(f'Epoch: {epoch} - Step: {batch} - Validation loss: {val_loss.result().numpy()} - Validation accuracy: {val_accuracy.result().numpy()*100}\n')
            # save weights
            checkpoint.save(os.path.join(ckpt_dir, 'ckpt'))
            model.save(os.path.join(args.output_dir, f'model-rnd-{args.rnd}'))
            with writer.as_default():
                tf.summary.scalar("val_loss", val_loss.result().numpy(), step=epoch)
                tf.summary.scalar("val_accuracy", val_accuracy.result().numpy(), step=epoch)
                writer.flush()
            vd_writer.write(f'{epoch}\t{batch}\t{val_loss.result().numpy()}\t{val_accuracy.result().numpy()}\n')

            # reset metrics variables
            val_loss.reset_states()
            train_accuracy.reset_states()
            val_accuracy.reset_states()

            define end of current epoch
            epoch += 1

    # all_labels = all_labels[0].numpy()
    # print(f'number of reads: {len(all_labels)}')
    # num_extra_reads = num_train_steps*args.batch_size - args.train_reads_per_epoch
    # print(f'number of extra reads: {num_extra_reads}')
    # all_labels = all_labels[:-num_extra_reads]
    # print(f'number of reads: {len(all_labels)}')
    # print(f'number of reads in set: {args.train_reads_per_epoch}')

    # d_labels = defaultdict(int)
    # for i in range(len(all_labels)):
    #     d_labels[all_labels[i]] += 1

    # print(d_labels)
    
    # if hvd.rank() == 0 and args.model_type != 'BERT':
    if args.model_type not in ['BERT', 'BERT_HUGGINGFACE']:
        # save final embeddings
        emb_weights = model.get_layer('embedding').get_weights()[0]
        out_v = io.open(os.path.join(args.output_dir, f'embeddings_rnd_{args.rnd}.tsv'), 'w', encoding='utf-8')
        print(f'# embeddings: {len(emb_weights)}')
        for i in range(len(emb_weights)):
            vec = emb_weights[i]
            out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_v.close()

    end = datetime.datetime.now()

    # if hvd.rank() == 0:
    total_time = end - start
    hours, seconds = divmod(total_time.seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    # create summary file
    #     if args.model_type != 'BERT':
    #         f.write(f'Vocabulary size\t{vocab_size}\nEmbedding size\t{args.embedding_size}\n')

    with open(os.path.join(args.output_dir, f'training-summary-rnd-{args.rnd}.tsv'), 'a') as f:
        f.write(f'Date\t{datetime.datetime.now().strftime("%d/%m/%Y")}\nTime\t{datetime.datetime.now().strftime("%H:%M:%S")}\n'
                f'Model\t{args.model_type}\nRound of training\t{args.rnd}\nEpochs\t{args.epochs}\n'
                f'Vector size\t{args.vector_size}\n'
                f'Dropout rate\t{args.dropout_rate}\nBatch size per gpu\t{args.batch_size}\n'
                f'Global batch size\t{args.batch_size}\nNumber of gpus\t{len(gpus)}\n'
                # f'Global batch size\t{args.batch_size*hvd.size()}\nNumber of gpus\t{hvd.size()}\n'
                f'Training set size\t{train_reads_per_epoch}\nValidation set size\t{val_reads_per_epoch}\n'
                f'Number of steps per epoch\t{nstep_per_epoch}\nNumber of steps for validation dataset\t{val_steps}\n'
                f'Initial learning rate\t{args.init_lr}\n')
        # \nNumber of classes\t{num_labels}
        f.write("\nTraining runtime:\t%02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))
    print("\nTraining runtime: %02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))
    td_writer.close()
    vd_writer.close()


if __name__ == "__main__":
    main()
