import datetime
import tensorflow as tf
import horovod.tensorflow as hvd
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import nvidia.dali.plugin.tf as dali_tf
from models import AlexNet
import os
import sys
import json
import glob
import time
import numpy as np
import math
from collections import defaultdict
import argparse

dl_toda_dir = '/'.join(os.path.dirname(os.path.abspath( __file__ )).split('/')[0:-1])

# disable eager execution
# tf.compat.v1.disable_eager_execution()
print(f'Is eager execution enabled: {tf.executing_eagerly()}')

# print which unit (CPU/GPU) is used for an operation
#tf.debugging.set_log_device_placement(True)

# enable XLA = XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that can accelerate
# TensorFlow models with potentially no source code changes
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
# use hvd.local_rank() for gpu pinning instead of hvd.rank()
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f'GPU RANK: {hvd.rank()}/{hvd.local_rank()} - LIST GPUs: {gpus}')
# comment next 2 lines if testing large dataset
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


# define the DALI pipeline
@pipeline_def
def get_dali_pipeline(tfrec_filenames, tfrec_idx_filenames, shard_id, num_gpus, dali_cpu=True, training=True):
    inputs = fn.readers.tfrecord(path=tfrec_filenames,
                                 index_path=tfrec_idx_filenames,
                                 random_shuffle=training,
                                 shard_id=shard_id,
                                 num_shards=num_gpus,
                                 # initial_fill=10000,
                                 features={
                                     "read": tfrec.VarLenFeature([], tfrec.int64, 0),
                                     "label": tfrec.FixedLenFeature([1], tfrec.int64, -1)})
    # retrieve reads and labels and copy them to the gpus
    reads = inputs["read"].gpu()
    labels = inputs["label"].gpu()
    return reads, labels


class DALIPreprocessor(object):
    def __init__(self, filenames, idx_filenames, batch_size, num_threads, dali_cpu=True,
               deterministic=False, training=False):

        device_id = hvd.local_rank()
        shard_id = hvd.rank()
        num_gpus = hvd.size()
        self.pipe = get_dali_pipeline(tfrec_filenames=filenames, tfrec_idx_filenames=idx_filenames, batch_size=batch_size,
                                      num_threads=num_threads, device_id=device_id, shard_id=shard_id, num_gpus=num_gpus,
                                      dali_cpu=dali_cpu, training=training, seed=7 * (1 + hvd.rank()) if deterministic else None)

        self.daliop = dali_tf.DALIIterator()

        self.batch_size = batch_size
        self.device_id = device_id

        self.dalidataset = dali_tf.DALIDataset(fail_on_device_mismatch=False, pipeline=self.pipe,
            output_shapes=((batch_size, 239), (batch_size)),
            batch_size=batch_size, output_dtypes=(tf.int64, tf.int64), device_id=device_id)

    def get_device_dataset(self):
        return self.dalidataset


@tf.function
def testing_step(data_type, reads, labels, model, loss=None, test_loss=None, test_accuracy=None):
    probs = model(reads, training=False)
    if data_type == 'test':
        test_accuracy.update_state(labels, probs)
        loss_value = loss(labels, probs)
        test_loss.update_state(loss_value)
    pred_labels = tf.math.argmax(probs, axis=1)
    pred_probs = tf.reduce_max(probs, axis=1)

    # return probs, pred_labels, pred_probs
    return pred_labels, pred_probs


def main():
    start = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords', type=str, help='path to tfrecords', required=True)
    parser.add_argument('--dali_idx', type=str, help='path to dali indexes files', required=True)
    parser.add_argument('--data_type', type=str, help='type of data tested', required=True, choices=['sim', 'meta'])
    parser.add_argument('--output_dir', type=str, help='directory to store results', default=os.getcwd())
    parser.add_argument('--epoch', type=int, help='epoch of checkpoint', default=14)
    parser.add_argument('--batch_size', type=int, help='batch size per gpu', default=8192)
    parser.add_argument('--model', type=str, help='path to directory containing model in SavedModel format')
    parser.add_argument('--ckpt', type=str, help='path to directory containing checkpoint file', required=('--epoch' in sys.argv))
    # parser.add_argument('--save_probs', help='save probability distributions', action='store_true')

    args = parser.parse_args()

    # define some training and model parameters
    vector_size = 250 - 12 + 1
    vocab_size = 8390657
    embedding_size = 60
    dropout_rate = 0.7

    # load class_mapping file mapping label IDs to species
    path_class_mapping = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/data/species_labels.json'
    f = open(path_class_mapping)
    class_mapping = json.load(f)
    num_classes = len(class_mapping)
    # create dtype policy
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    # define metrics
    if args.data_type == 'sim':
        loss = tf.losses.SparseCategoricalCrossentropy()
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    init_lr = 0.0001
    opt = tf.keras.optimizers.Adam(init_lr)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    if hvd.rank() == 0:
        # create output directories
        if not os.path.isdir(args.output_dir):
            os.makedirs(os.path.join(args.output_dir))

    # load model
    if args.ckpt is not None:
        model = AlexNet(args, vector_size, embedding_size, num_classes, vocab_size, dropout_rate)
        checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
        checkpoint.restore(f'{args.ckpt}-{args.epoch}').expect_partial()
    elif args.model is not None:
        model = tf.keras.models.load_model(args.model, 'model')
            # restore the last checkpointed values to the model
    #        checkpoint = tf.train.Checkpoint(model)
    #        checkpoint.restore(tf.train.latest_checkpoint(os.path.join(input_dir, f'run-{run_num}', 'ckpts')))
    #        ckpt_path = os.path.join(input_dir, f'run-{run_num}', 'ckpts/ckpts')
    #        latest_ckpt = tf.train.latest_checkpoint(os.path.join(input_dir, f'run-{run_num}', 'ckpts'))
    #        print(f'latest ckpt: {latest_ckpt}')
    #        model.load_weights(os.path.join(input_dir, f'run-{run_num}', f'ckpts/ckpts-{epoch}'))

    # get list of testing tfrecords and number of reads per tfrecords
    test_files = sorted(glob.glob(os.path.join(args.tfrecords, '*.tfrec')))
    test_idx_files = sorted(glob.glob(os.path.join(args.dali_idx, '*.idx')))
    num_reads_files = sorted(glob.glob(os.path.join(args.tfrecords, '*-read_count')))
    read_ids_files = sorted(glob.glob(os.path.join(args.tfrecords, '*-read_ids.tsv'))) if args.data_type == 'meta' else []

    # split tfrecords between gpus
    test_files_per_gpu = len(test_files)//hvd.size()

    if hvd.rank() != hvd.size() - 1:
        gpu_test_files = test_files[hvd.rank()*test_files_per_gpu:(hvd.rank()+1)*test_files_per_gpu]
        gpu_test_idx_files = test_idx_files[hvd.rank()*test_files_per_gpu:(hvd.rank()+1)*test_files_per_gpu]
        gpu_num_reads_files = num_reads_files[hvd.rank()*test_files_per_gpu:(hvd.rank()+1)*test_files_per_gpu]
        gpu_read_ids_files = read_ids_files[hvd.rank()*test_files_per_gpu:(hvd.rank()+1)*test_files_per_gpu] if len(read_ids_files) != 0 else None
    else:
        gpu_test_files = test_files[hvd.rank()*test_files_per_gpu:len(test_files)]
        gpu_test_idx_files = test_idx_files[hvd.rank()*test_files_per_gpu:len(test_files)]
        gpu_num_reads_files = num_reads_files[hvd.rank()*test_files_per_gpu:len(test_files)]
        gpu_read_ids_files = read_ids_files[hvd.rank()*test_files_per_gpu:len(test_files)] if len(read_ids_files) != 0 else None

    print(gpu_test_files)

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

        num_preprocessing_threads = 4
        test_preprocessor = DALIPreprocessor(gpu_test_files[i], gpu_test_idx_files[i], args.batch_size, num_preprocessing_threads, dali_cpu=True, deterministic=False, training=False)

        test_input = test_preprocessor.get_device_dataset()

        # create empty arrays to store the predicted and true values, the confidence scores and the probability distributions
        # all_predictions = tf.zeros([args.batch_size, NUM_CLASSES], dtype=tf.dtypes.float32, name=None)
        all_pred_sp = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]
        all_prob_sp = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]
        all_labels = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]
        for batch, (reads, labels) in enumerate(test_input.take(test_steps), 1):
            if args.data_type == 'meta':
                # batch_predictions, batch_pred_sp, batch_prob_sp = testing_step(args.data_type, reads, labels, model)
                batch_pred_sp, batch_prob_sp = testing_step(args.data_type, reads, labels, model)
            elif args.data_type == 'sim':
                # batch_predictions, batch_pred_sp, batch_prob_sp = testing_step(args.data_type, reads, labels, model, loss, test_loss, test_accuracy)
                batch_pred_sp, batch_prob_sp = testing_step(args.data_type, reads, labels, model, loss, test_loss, test_accuracy)

            if batch == 1:
                all_labels = [labels]
                all_pred_sp = [batch_pred_sp]
                all_prob_sp = [batch_prob_sp]
                # all_predictions = batch_predictions
            else:
                # all_predictions = tf.concat([all_predictions, batch_predictions], 0)
                all_pred_sp = tf.concat([all_pred_sp, [batch_pred_sp]], 1)
                all_prob_sp = tf.concat([all_prob_sp, [batch_prob_sp]], 1)
                all_labels = tf.concat([all_labels, [labels]], 1)

        # get list of true species, predicted species and predicted probabilities
        # all_predictions = all_predictions.numpy()
        all_pred_sp = all_pred_sp[0].numpy()
        all_prob_sp = all_prob_sp[0].numpy()
        all_labels = all_labels[0].numpy()

        # adjust the list of predicted species and read ids if necessary
        if len(all_labels) > num_reads:
            num_extra_reads = (test_steps*args.batch_size) - num_reads
            # all_predictions = all_predictions[:-num_extra_reads]
            all_pred_sp = all_pred_sp[:-num_extra_reads]
            all_prob_sp = all_prob_sp[:-num_extra_reads]
            all_labels = all_labels[:-num_extra_reads]

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
                    out_f.write(f'{all_labels[j]}\t{all_pred_sp[j]}\t{all_prob_sp[j]}\n')
            # if args.save_probs:
            #     # save predictions and labels to file
            #     np.save(os.path.join(args.output_dir, f'{gpu_test_files[i].split("/")[-1].split(".")[0]}-prob-out.npy'), all_predictions)
            #     np.save(os.path.join(args.output_dir, f'{gpu_test_files[i].split("/")[-1].split(".")[0]}-labels-out.npy'), all_labels)

        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)

    end = datetime.datetime.now()
    total_time = end - start
    hours, seconds = divmod(total_time.seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    with open(os.path.join(args.output_dir, f'testing-summary-{hvd.rank()}.tsv'), 'w') as outfile:
        outfile.write(f'{hvd.rank()}\t{args.batch_size}\t{hvd.size()}\t{hvd.rank()}\t{len(gpu_test_files)}\t{num_reads_classified}\t')
        if args.data_type == 'sim':
            outfile.write(f'{test_accuracy.result().numpy()}\t{test_loss.result().numpy()}\t')
        if args.ckpt:
            outfile.write(f'{args.epoch}')
        else:
            outfile.write(f'model saved at last epoch')
        outfile.write(f'\t{hours}:{minutes}:{seconds}:{total_time.microseconds}\t')

        if len(elapsed_time) > 1:
            outfile.write(f'{(num_reads_classified / elapsed_time.sum())} reads/sec\n')
        else:
            outfile.write(f'{(num_reads_classified / elapsed_time[0])} reads/sec\n')


if __name__ == "__main__":
    main()
