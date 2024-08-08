import datetime
import tensorflow as tf
import horovod.tensorflow as hvd
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
from BERT import BertConfig, BertModel
import os
import sys
import json
import glob
import time
# import numpy as np
import math
import argparse


# set seed
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)
tf.experimental.numpy.random.seed(seed)


dl_toda_dir = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])

# disable eager execution
# tf.compat.v1.disable_eager_execution()
print(f'Is eager execution enabled: {tf.executing_eagerly()}')

# print which unit (CPU/GPU) is used for an operation
#tf.debugging.set_log_device_placement(True)

# enable XLA = XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that can accelerate
# TensorFlow models with potentially no source code changes
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# # Initialize Horovod
# hvd.init()

# # Pin GPU to be used to process local rank (one GPU per process)
# # use hvd.local_rank() for gpu pinning instead of hvd.rank()
# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(f'GPU RANK: {hvd.rank()}/{hvd.local_rank()} - LIST GPUs: {gpus}')
# # comment next 2 lines if testing large dataset
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# if gpus:
#     tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


# define the DALI pipeline
# @pipeline_def
# def get_dali_pipeline(tfrec_filenames, tfrec_idx_filenames, shard_id, num_gpus, dali_cpu=True, training=True):
#     inputs = fn.readers.tfrecord(path=tfrec_filenames,
#                                  index_path=tfrec_idx_filenames,
#                                  random_shuffle=training,
#                                  shard_id=shard_id,
#                                  num_shards=num_gpus,
#                                  initial_fill=10000,
#                                  features={
#                                      "read": tfrec.VarLenFeature([], tfrec.int64, 0),
#                                      "label": tfrec.FixedLenFeature([1], tfrec.int64, -1)})
#     # retrieve reads and labels and copy them to the gpus
#     reads = inputs["read"].gpu()
#     labels = inputs["label"].gpu()
#     return reads, labels


# class DALIPreprocessor(object):
#     def __init__(self, filenames, idx_filenames, batch_size, num_threads, dali_cpu=True,
#                deterministic=False, training=False):
#
#         device_id = hvd.local_rank()
#         shard_id = hvd.rank()
#         num_gpus = hvd.size()
#         self.pipe = get_dali_pipeline(tfrec_filenames=filenames, tfrec_idx_filenames=idx_filenames, batch_size=batch_size,
#                                       num_threads=num_threads, device_id=device_id, shard_id=shard_id, num_gpus=num_gpus,
#                                       dali_cpu=dali_cpu, training=training, seed=7 * (1 + hvd.rank()) if deterministic else None)
#
#         self.daliop = dali_tf.DALIIterator()
#
#         self.batch_size = batch_size
#         self.device_id = device_id
#
#         self.dalidataset = dali_tf.DALIDataset(fail_on_device_mismatch=False, pipeline=self.pipe,
#             output_shapes=((batch_size, 239), (batch_size)),
#             batch_size=batch_size, output_dtypes=(tf.int64, tf.int64), device_id=device_id)
#
#     def get_device_dataset(self):
#         return self.dalidataset

# class DALIPreprocessor(object):
#     def __init__(self, filenames, idx_filenames, batch_size, num_threads, vector_size, dali_cpu=True,
#                deterministic=False, training=False):

#         device_id = hvd.local_rank()
#         shard_id = hvd.rank()
#         num_gpus = hvd.size()
#         self.pipe = get_dali_pipeline(tfrec_filenames=filenames, tfrec_idx_filenames=idx_filenames, batch_size=batch_size,
#                                       num_threads=num_threads, device_id=device_id, shard_id=shard_id, num_gpus=num_gpus,
#                                       dali_cpu=dali_cpu, training=training, seed=7 * (1 + hvd.rank()) if deterministic else None)

#         self.daliop = dali_tf.DALIIterator()

#         self.batch_size = batch_size
#         self.device_id = device_id

#         self.dalidataset = dali_tf.DALIDataset(fail_on_device_mismatch=False, pipeline=self.pipe,
#             output_shapes=((batch_size, vector_size), (batch_size)),
#             batch_size=batch_size, output_dtypes=(tf.int64, tf.int64), device_id=device_id)

#     def get_device_dataset(self):
#         return self.dalidataset


# define the DALI pipeline
@pipeline_def
def get_dali_pipeline(tfrec_filenames, tfrec_idx_filenames, shard_id, initial_fill, num_gpus, training=True):
    # prefetch_queue_depth = 100
    # read_ahead = True
    stick_to_shard = True
    inputs = fn.readers.tfrecord(path=tfrec_filenames,
                                 index_path=tfrec_idx_filenames,
                                 random_shuffle=training,
                                 shard_id=shard_id,
                                 num_shards=num_gpus,
                                 initial_fill=initial_fill,
                                 # prefetch_queue_depth=prefetch_queue_depth,
                                 # read_ahead=read_ahead,
                                 stick_to_shard=stick_to_shard,
                                 features={
                                     "read": tfrec.VarLenFeature([], tfrec.int64, 0),
                                     "label": tfrec.FixedLenFeature([1], tfrec.int64, -1)})
    # retrieve reads and labels and copy them to the gpus
    reads = inputs["read"].gpu()
    labels = inputs["label"].gpu()
    return (reads, labels)


# define the BERT DALI pipeline
@pipeline_def
def get_bert_dali_pipeline(tfrec_filenames, tfrec_idx_filenames, shard_id, initial_fill, num_gpus, training=True):
    inputs = fn.readers.tfrecord(path=tfrec_filenames,
                                 index_path=tfrec_idx_filenames,
                                 random_shuffle=training,
                                 shard_id=0,
                                 num_shards=1,
                                 stick_to_shard=False,
                                 initial_fill=initial_fill,
                                 features={
                                     "input_ids": tfrec.VarLenFeature([], tfrec.int64, 0),
                                     "input_mask": tfrec.VarLenFeature([], tfrec.int64, 0),
                                     "segment_ids": tfrec.VarLenFeature([], tfrec.int64, 0),
                                     "is_real_example": tfrec.FixedLenFeature([1], tfrec.int64, -1),
                                     "label_ids": tfrec.FixedLenFeature([1], tfrec.int64, -1)})
    # retrieve reads and labels and copy them to the gpus
    input_ids = inputs["input_ids"].gpu()
    input_mask = inputs["input_mask"].gpu()
    segment_ids = inputs["segment_ids"].gpu()
    label_ids = inputs['label_ids'].gpu()
    is_real_example = inputs['is_real_example'].gpu()

    return (input_ids, input_mask, segment_ids, label_ids, is_real_example)


class DALIPreprocessor(object):
    def __init__(self, args, filenames, idx_filenames, batch_size, vector_size, initial_fill, deterministic=False, training=False):

        device_id = hvd.local_rank()
        shard_id = hvd.rank()
        num_gpus = hvd.size()
        
        self.batch_size = batch_size
        self.device_id = device_id

        if args.model_type == "BERT":
            self.pipe = get_bert_dali_pipeline(tfrec_filenames=filenames, tfrec_idx_filenames=idx_filenames, batch_size=batch_size,
                                      device_id=device_id, shard_id=shard_id, initial_fill=initial_fill, num_gpus=num_gpus,
                                      training=training, seed=7 * (1 + hvd.rank()) if deterministic else None)

            self.dalidataset = dali_tf.DALIDataset(fail_on_device_mismatch=False, pipeline=self.pipe,
                output_shapes=((args.batch_size, vector_size), (args.batch_size, vector_size), (args.batch_size, vector_size), (args.batch_size), (args.batch_size)),
                batch_size=batch_size, output_dtypes=(tf.int64, tf.int64, tf.int64, tf.int64, tf.int64), device_id=device_id)
        else:
            self.pipe = get_dali_pipeline(tfrec_filenames=filenames, tfrec_idx_filenames=idx_filenames, batch_size=batch_size,
                                      device_id=device_id, shard_id=shard_id, initial_fill=initial_fill, num_gpus=num_gpus,
                                      training=training, seed=7 * (1 + hvd.rank()) if deterministic else None)
   
            self.dalidataset = dali_tf.DALIDataset(fail_on_device_mismatch=False, pipeline=self.pipe,
                output_shapes=((batch_size, vector_size), (batch_size)),
                batch_size=batch_size, output_dtypes=(tf.int64, tf.int64), device_id=device_id)

    def get_device_dataset(self):
        return self.dalidataset


def build_dataset(filenames, batch_size, vector_size, num_classes, datatype, is_training, drop_remainder):

    def load_tfrecords_with_reads(proto_example):
        data_description = {
            'read': tf.io.VarLenFeature(tf.int64),
            # 'read': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
        }
        # load one example
        parsed_example = tf.io.parse_single_example(serialized=proto_example, features=data_description)
        read = parsed_example['read']
        label = tf.cast(parsed_example['label'], tf.int64)
        read = tf.sparse.to_dense(read)
        return read, label


    def load_tfrecords_with_sentences(proto_example):
        name_to_features = {
          "input_ids": tf.io.FixedLenFeature([vector_size], tf.int64),
          "input_mask": tf.io.FixedLenFeature([vector_size], tf.int64),
          "segment_ids": tf.io.FixedLenFeature([vector_size], tf.int64),
          "label_ids": tf.io.FixedLenFeature([], tf.int64),
          "is_real_example": tf.io.FixedLenFeature([], tf.int64),
        }
        # load one example
        parsed_example = tf.io.parse_single_example(serialized=proto_example, features=name_to_features)
        input_ids = parsed_example['input_ids']
        input_mask = parsed_example['input_mask']
        segment_ids = parsed_example['segment_ids']
        label_ids = parsed_example['label_ids']
        is_real_example = parsed_example['is_real_example']

        return  (input_ids, input_mask, segment_ids, label_ids, is_real_example)

    """ Return data in TFRecords """
    fn_load_data = {'reads': load_tfrecords_with_reads, 'sentences': load_tfrecords_with_sentences}

    dataset = tf.data.TFRecordDataset([filenames])

    if is_training:
        dataset = dataset.repeat()
        # dataset = dataset.shuffle(buffer_size=100)

    dataset = dataset.map(map_func=fn_load_data[datatype])
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)


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
def testing_step(data_type, model_type, data, model, loss=None, test_loss=None, test_accuracy=None, target_label=None):
    training = False
    if model_type == 'BERT':
        input_ids, input_mask, token_type_ids, labels, is_real_example = data
        probs, log_probs, logits = model(input_ids, input_mask, token_type_ids, training)
    else:
        reads, labels = data
        probs = model(reads, training=training)
    
    if data_type == 'sim':
        test_accuracy.update_state(labels, probs)
        loss_value = loss(labels, probs)
        test_loss.update_state(loss_value)

    # get predicted labels and confidence scores
    pred_labels = tf.math.argmax(probs, axis=1)
    if tf.shape(probs)[1] == 2:
        pred_probs = probs
    else:
        pred_probs = tf.reduce_max(probs, axis=1)

    if target_label:
        label_prob = tf.gather(probs, target_label, axis=1)

    return probs, pred_labels, pred_probs, labels
    # return pred_labels, pred_probs, label_prob


def main():
    start = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords', type=str, help='path to tfrecords', required=True)
    parser.add_argument('--dali_idx', type=str, help='path to dali indexes files')
    parser.add_argument('--data_type', type=str, help='type of data tested', required=True, choices=['sim', 'meta'])
    parser.add_argument('--output_dir', type=str, help='directory to store results', default=os.getcwd())
    parser.add_argument('--init_lr', type=float, help='initial learning rate', default=0.0001)
    # parser.add_argument('--epoch', type=int, help='epoch of checkpoint', default=14)
    parser.add_argument('--batch_size', type=int, help='batch size per gpu', default=8192)
    parser.add_argument('--DNA_model', action='store_true', default=False)
    parser.add_argument('--n_rows', type=int, default=50)
    parser.add_argument('--n_cols', type=int, default=5)
    parser.add_argument('--kh_conv_1', type=int, default=2)
    parser.add_argument('--kh_conv_2', type=int, default=2)
    parser.add_argument('--kw_conv_1', type=int, default=3)
    parser.add_argument('--kw_conv_2', type=int, default=4)
    parser.add_argument('--sh_conv_1', type=int, default=1)
    parser.add_argument('--sw_conv_1', type=int, default=1)
    parser.add_argument('--sh_conv_2', type=int, default=1)
    parser.add_argument('--sw_conv_2', type=int, default=1)
    parser.add_argument('--num_labels', type=int, help='number of labels', default=2)
    parser.add_argument('--nvidia_dali', action='store_true', default=False, required=('val_idx_files' in sys.argv and 'train_idx_files' in sys.argv))
    parser.add_argument('--k_value', type=int, help='length of kmer strings', default=12)
    parser.add_argument('--target_label', type=int, help='output prediction scores of target label')
    parser.add_argument('--labels', type=int, help='number of labels')
    parser.add_argument('--embedding_size', type=int, help='size of embedding vectors', default=60)
    parser.add_argument('--dropout_rate', type=float, help='dropout rate to apply to layers', default=0.7)
    parser.add_argument('--vector_size', type=int, help='size of input vectors')
    parser.add_argument('--vocab', help="Path to the vocabulary file", required=('AlexNet' in sys.argv))
    parser.add_argument('--model_type', type=str, help='type of model', choices=['DNA_1', 'DNA_2', 'AlexNet', 'VGG16', 'VDCNN', 'LSTM', 'BERT'])
    parser.add_argument('--bert_config_file', type=str, help='path to bert config file', required=('BERT' in sys.argv))
    parser.add_argument('--model', type=str, help='path to directory containing model in SavedModel format')
    parser.add_argument('--class_mapping', type=str, help='path to json file containing dictionary mapping taxa to labels', default=os.path.join(dl_toda_dir, 'data', 'species_labels.json'))
    parser.add_argument('--ckpt', type=str, help='path to directory containing checkpoint file', required=('--epoch' in sys.argv))
    parser.add_argument('--max_read_size', type=int, help='maximum read size in training dataset', default=250)
    parser.add_argument('--initial_fill', type=int, help='size of the buffer for random shuffling', default=10000)
    # parser.add_argument('--save_probs', help='save probability distributions', action='store_true')
    args = parser.parse_args()

    # Initialize Horovod
    hvd.init()
    # Map one GPU per process
    # use hvd.local_rank() for gpu pinning instead of hvd.rank()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'GPU RANK: {hvd.rank()}/{hvd.local_rank()} - LIST GPUs: {gpus}')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    models = {'DNA_1': DNA_net_1, 'DNA_2': DNA_net_2, 'AlexNet': AlexNet, 'VGG16': VGG16, 'VDCNN': VDCNN, 'LSTM': LSTM, 'BERT': BertModel}

    # get vocabulary size
    if args.model_type != 'BERT':
        with open(args.vocab, 'r') as f:
            content = f.readlines()
            vocab_size = len(content)

    # load class_mapping file mapping label IDs to species
    if args.class_mapping:
        f = open(args.class_mapping)
        class_mapping = json.load(f)
        num_labels = len(class_mapping)
    else:
        num_labels = args.num_labels

    # # create dtype policy
    # policy = tf.keras.mixed_precision.Policy('mixed_float16')
    # tf.keras.mixed_precision.set_global_policy(policy)
    # print('Compute dtype: %s' % policy.compute_dtype)
    # print('Variable dtype: %s' % policy.variable_dtype)
    # print(f'2: {datetime.datetime.now()}')
    
    # define metrics
    if args.data_type == 'sim':
        loss = tf.losses.SparseCategoricalCrossentropy()
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    if hvd.rank() == 0:
        # create output directories
        if not os.path.isdir(args.output_dir):
            os.makedirs(os.path.join(args.output_dir))

    # get list of testing tfrecords and number of reads per tfrecords
    test_files = sorted(glob.glob(os.path.join(args.tfrecords, '*.tfrec')))
    num_reads_files = sorted(glob.glob(os.path.join(args.tfrecords, '*-read_count')))
    read_ids_files = sorted(glob.glob(os.path.join(args.tfrecords, '*-read_ids.tsv'))) if args.data_type == 'meta' else []

    if args.nvidia_dali:
        # get nvidia dali indexes
        test_idx_files = sorted(glob.glob(os.path.join(args.dali_idx, '*.idx')))
    
    # split tfrecords between gpus
    test_files_per_gpu = len(test_files)//hvd.size()

    if hvd.rank() != hvd.size() - 1:
        gpu_test_files = test_files[hvd.rank()*test_files_per_gpu:(hvd.rank()+1)*test_files_per_gpu]
        gpu_num_reads_files = num_reads_files[hvd.rank()*test_files_per_gpu:(hvd.rank()+1)*test_files_per_gpu]
        gpu_read_ids_files = read_ids_files[hvd.rank()*test_files_per_gpu:(hvd.rank()+1)*test_files_per_gpu] if len(read_ids_files) != 0 else None

        if args.nvidia_dali:
            gpu_test_idx_files = test_idx_files[hvd.rank()*test_files_per_gpu:(hvd.rank()+1)*test_files_per_gpu]
    else:
        gpu_test_files = test_files[hvd.rank()*test_files_per_gpu:len(test_files)]
        gpu_num_reads_files = num_reads_files[hvd.rank()*test_files_per_gpu:len(test_files)]
        gpu_read_ids_files = read_ids_files[hvd.rank()*test_files_per_gpu:len(test_files)] if len(read_ids_files) != 0 else None

        if args.nvidia_dali:
            gpu_test_idx_files = test_idx_files[hvd.rank()*test_files_per_gpu:len(test_files)]

    init_lr = args.init_lr
    opt = tf.keras.optimizers.Adam(init_lr)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)


    # load model
    if args.ckpt is not None:
        if args.model_type == 'BERT':
            config = BertConfig.from_json_file(args.bert_config_file)
            model = BertModel(config=config)
        else:
            model = models[args.model_type](args, args.vector_size, args.embedding_size, num_classes, vocab_size, args.dropout_rate)
        checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
        # checkpoint.restore(os.path.join(args.ckpt, f'ckpt-{args.epoch}')).expect_partial()
        checkpoint.restore(args.ckpt).expect_partial()
    elif args.model is not None:
        model = tf.keras.models.load_model(args.model, 'model')
            # restore the last checkpointed values to the model
    #        checkpoint = tf.train.Checkpoint(model)
    #        checkpoint.restore(tf.train.latest_checkpoint(os.path.join(input_dir, f'run-{run_num}', 'ckpts')))
    #        ckpt_path = os.path.join(input_dir, f'run-{run_num}', 'ckpts/ckpts')
    #        latest_ckpt = tf.train.latest_checkpoint(os.path.join(input_dir, f'run-{run_num}', 'ckpts'))
    #        print(f'latest ckpt: {latest_ckpt}')
    #        model.load_weights(os.path.join(input_dir, f'run-{run_num}', f'ckpts/ckpts-{epoch}'))


    elapsed_time = []
    num_reads_classified = 0
    for i in range(len(gpu_test_files)):
        print(gpu_test_files[i])
        start_time = time.time()
        # get number of reads in test file
        with open(os.path.join(args.tfrecords, gpu_num_reads_files[i]), 'r') as f:
            num_reads = int(f.readline())
        num_reads_classified += num_reads

        # compute number of steps required to iterate over entire test set
        test_steps = math.ceil(num_reads/(args.batch_size))

        # load data
        if args.nvidia_dali:
            test_preprocessor = DALIPreprocessor(args, gpu_test_files[i], gpu_test_idx_files[i], args.batch_size, args.vector_size, args.initial_fill, deterministic=False, training=False)

            test_input = test_preprocessor.get_device_dataset()
        else:
            if args.model_type == 'BERT':
                datatype = 'sentences'
            else:
                datatype = 'reads'

            test_input = build_dataset(gpu_test_files[i], args.batch_size, args.vector_size, num_labels, datatype, is_training=False, drop_remainder=True)


        # create empty arrays to store the predicted and true values, the confidence scores and the probability distributions
        # all_predictions = tf.zeros([args.batch_size, NUM_CLASSES], dtype=tf.dtypes.float32, name=None)
        all_pred_sp = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]
        all_prob_sp = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]
        all_labels = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]
        # all_prob_labels = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]
        for batch, data in enumerate(test_input.take(test_steps), 1):
            if args.data_type == 'meta':
                # batch_predictions, batch_pred_sp, batch_prob_sp = testing_step(args.data_type, reads, labels, model)
                batch_pred_sp, batch_prob_sp = testing_step(args.data_type, reads, labels, model)
            elif args.data_type == 'sim':
                # batch_predictions, batch_pred_sp, batch_prob_sp = testing_step(args.data_type, reads, labels, model, loss, test_loss, test_accuracy)
                # batch_pred_sp, batch_prob_sp, batch_label_prob = testing_step(args.data_type, reads, labels, model, loss, test_loss, test_accuracy, args.target_label)
                batch_predictions, batch_pred_sp, batch_prob_sp, labels = testing_step(args.data_type, args.model_type, data, model, loss, test_loss, test_accuracy)
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

        # adjust the list of predicted species and read ids if necessary
        if len(all_labels) > num_reads:
            num_extra_reads = (test_steps*args.batch_size) - num_reads
            # all_predictions = all_predictions[:-num_extra_reads]
            all_pred_sp = all_pred_sp[:-num_extra_reads]
            all_prob_sp = all_prob_sp[:-num_extra_reads]
            all_labels = all_labels[:-num_extra_reads]
            # all_prob_labels = all_prob_labels[:-num_extra_reads]

        if args.data_type == 'meta':
            # get dictionary mapping read ids to labels
            with open(os.path.join(args.tfrecords, gpu_read_ids_files[i]), 'r') as f:
                content = f.readlines()
                dict_read_ids = {content[j].rstrip().split('\t')[1]: '@' + content[j].rstrip().split('\t')[0] for j in range(len(content))}
            # write results to file
            with open(os.path.join(args.output_dir, f'{gpu_test_files[i].split("/")[-1].split(".")[0]}-out.tsv'), 'w') as out_f:
                for j in range(num_reads):
                    out_f.write(f'{dict_read_ids[str(all_labels[j])]}\t\t{all_pred_sp[j]}\t{all_prob_sp[j]}\n')

        elif args.data_type == 'sim':
            # write results to file
            out_filename = os.path.join(args.output_dir, f'{gpu_test_files[i].split("/")[-1].split(".")[0]}-out.tsv') if len(gpu_test_files[i].split("/")[-1].split(".")) == 2 else os.path.join(args.output_dir, f'{".".join(gpu_test_files[i].split("/")[-1].split(".")[0:2])}-out.tsv')
            with open(out_filename, 'w') as out_f:
                for j in range(num_reads):
                    out_f.write(f'{all_labels[j]}\t{all_pred_sp[j]}\t')
                    # out_f.write(f'{all_labels[j]}\t{all_pred_sp[j]}\t{all_prob_sp[j]}\t{all_prob_labels[j]}\n')
                    if len(all_prob_sp[j]) == args.labels:
                        out_f.write(f'{all_prob_sp[j][0]}\t{all_prob_sp[j][1]}\n')
                    else:
                        out_f.write(f'{all_prob_sp[j]}\n')
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

    with open(os.path.join(args.output_dir, f'testing-summary-{hvd.rank()}.tsv'), 'w') as outfile:
        outfile.write(f'{hvd.rank()}\t{args.batch_size}\t{hvd.size()}\t{hvd.rank()}\t{len(gpu_test_files)}\t{num_reads_classified}\t')
        if args.data_type == 'sim':
            outfile.write(f'{test_accuracy.result().numpy()}\t{test_loss.result().numpy()}\t')
        if args.ckpt:
            outfile.write(f'{args.ckpt}')
        # else:
        #     outfile.write(f'model saved at last epoch')
        outfile.write(f'\t{hours}:{minutes}:{seconds}:{total_time.microseconds}\t')

        if len(elapsed_time) > 1:
            outfile.write(f'{(num_reads_classified / sum(elapsed_time))} reads/sec\n')
        else:
            outfile.write(f'{(num_reads_classified / elapsed_time[0])} reads/sec\n')


if __name__ == "__main__":
    main()
