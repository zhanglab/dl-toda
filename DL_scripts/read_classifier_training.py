import tensorflow as tf
import horovod.tensorflow as hvd
import tensorflow.keras as keras
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import nvidia.dali.plugin.tf as dali_tf
import os
import sys
import json
import glob
import datetime
import numpy as np
import math
import io
import random
from models import AlexNet
from DNA_model import DNA_net
import argparse

dl_toda_dir = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])

# disable eager execution
#tf.compat.v1.disable_eager_execution()
print(f'Is eager execution enabled: {tf.executing_eagerly()}')

# print which unit (CPU/GPU) is used for an operation
#tf.debugging.set_log_device_placement(True)

# enable XLA = XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that can accelerate
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

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

# define the DALI pipeline
@pipeline_def
def get_dali_pipeline(tfrec_filenames, tfrec_idx_filenames, shard_id, num_gpus, dali_cpu=True, training=True):
    inputs = fn.readers.tfrecord(path=tfrec_filenames,
                                 index_path=tfrec_idx_filenames,
                                 random_shuffle=training,
                                 shard_id=shard_id,
                                 num_shards=num_gpus,
                                 initial_fill=10000,
                                 features={
                                     "read": tfrec.VarLenFeature([], tfrec.int64, 0),
                                     "label": tfrec.FixedLenFeature([1], tfrec.int64, -1)})
    # retrieve reads and labels and copy them to the gpus
    reads = inputs["read"].gpu()
    labels = inputs["label"].gpu()
    return (reads, labels)

class DALIPreprocessor(object):
    def __init__(self, filenames, idx_filenames, batch_size, num_threads, vector_size, dali_cpu=True,
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
            output_shapes=((batch_size, vector_size), (batch_size)),
            batch_size=batch_size, output_dtypes=(tf.int64, tf.int64), device_id=device_id)

    def get_device_dataset(self):
        return self.dalidataset

@tf.function
def training_step(reads, labels, train_accuracy, loss, opt, model, first_batch):
    with tf.GradientTape() as tape:
        probs = model(reads, training=True)
        # get the loss
        loss_value = loss(labels, probs)
        # scale the loss (multiply the loss by a factor) to avoid numeric underflow
        scaled_loss = opt.get_scaled_loss(loss_value)
    # use DistributedGradientTape to wrap tf.GradientTape and use an allreduce to
    # combine gradient values before applying gradients to model weights
    tape = hvd.DistributedGradientTape(tape)
    # get the scaled gradients
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    # get the unscaled gradients
    grads = opt.get_unscaled_gradients(scaled_gradients)
    #grads = tape.gradient(loss_value, model.trainable_variables)
    #opt.apply_gradients(zip(grads, model.trainable_variables))
    opt.apply_gradients(zip(grads, model.trainable_variables))
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    if first_batch:
        print(f'First_batch: {first_batch}')
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)

    #update training accuracy
    train_accuracy.update_state(labels, probs)

    return loss_value, grads

@tf.function
def testing_step(reads, labels, loss, val_loss, val_accuracy, model):
    probs = model(reads, training=False)
    val_accuracy.update_state(labels, probs)
    loss_value = loss(labels, probs)
    val_loss.update_state(loss_value)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords', type=str, help='path to tfrecords', required=True)
    parser.add_argument('--idx_files', type=str, help='path to dali index files', required=True)
    parser.add_argument('--class_mapping', type=str, help='path to json file containing dictionary mapping taxa to labels', default=os.path.join(dl_toda_dir, 'data', 'species_labels.json'))
    parser.add_argument('--output_dir', type=str, help='path to store model', default=os.getcwd())
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--epoch_to_resume', type=int, required=('-resume' in sys.argv))
    parser.add_argument('--n_rows', type=int, required=('--DNA_model' in sys.argv), default=50)
    parser.add_argument('--n_cols', type=int, required=('--DNA_model' in sys.argv), default=5)
    parser.add_argument('--kh_conv_1', type=int, required=('--DNA_model' in sys.argv), default=2)
    parser.add_argument('--kh_conv_2', type=int, required=('--DNA_model' in sys.argv), default=2)
    parser.add_argument('--kw_conv_1', type=int, required=('--DNA_model' in sys.argv), default=3)
    parser.add_argument('--kw_conv_2', type=int, required=('--DNA_model' in sys.argv), default=4)
    parser.add_argument('--ckpt', type=str, help='path to checkpoint file', required=('-resume' in sys.argv))
    parser.add_argument('--model', type=str, help='path to model', required=('-resume' in sys.argv))
    parser.add_argument('--epochs', type=int, help='number of epochs', default=30)
    parser.add_argument('--dropout_rate', type=float, help='dropout rate to apply to layers', default=0.7)
    parser.add_argument('--batch_size', type=int, help='batch size per gpu', default=512)
    parser.add_argument('--max_read_size', type=int, help='maximum read size in training dataset', default=250)
    parser.add_argument('--k_value', type=int, help='length of kmer strings', default=12)
    parser.add_argument('--embedding_size', type=int, help='size of embedding vectors', default=60)
    parser.add_argument('--rnd', type=int, help='round of training', default=1)
    parser.add_argument('--DNA_model', action='store_true', default=False)
    parser.add_argument('--num_train_samples', type=int, help='number of reads in training set', required=True)
    parser.add_argument('--num_val_samples', type=int, help='number of reads in validation set', required=True)
    parser.add_argument('--init_lr', type=float, help='initial learning rate', default=0.0001)
    parser.add_argument('--lr_decay', type=int, help='number of epochs before dividing learning rate in half', default=20)
    args = parser.parse_args()

    # define some training and model parameters
    if args.DNA_model:
        vector_size = 250
        vocab_size = 5
        model = 'DNANet'
    else:
        vector_size = args.max_read_size - args.kmer_size + 1
        vocab_size = ((4 ** args.k_value + 4 ** (args.k_value / 2)) / 2) + 1 if args.k_value % 2 == 0 else ((4 ** args.k_value) / 2) + 1
        model = 'AlexNet'

    # load class_mapping file mapping label IDs to species
    f = open(args.class_mapping)
    class_mapping = json.load(f)
    num_classes = len(class_mapping)

    # create dtype policy
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)

    # Get training and validation tfrecords
    train_files = sorted(glob.glob(os.path.join(args.tfrecords, 'train*.tfrec')))
    train_idx_files = sorted(glob.glob(os.path.join(args.idx_files, 'train*.idx')))
    val_files = sorted(glob.glob(os.path.join(args.tfrecords, 'val*.tfrec')))
    val_idx_files = sorted(glob.glob(os.path.join(args.idx_files, 'val*.idx')))
    # compute number of steps/batches per epoch
    nstep_per_epoch = args.num_train_samples // (args.batch_size*hvd.size())
    # compute number of steps/batches to iterate over entire validation set
    val_steps = args.num_val_samples // (args.batch_size*hvd.size())

    num_preprocessing_threads = 4
    train_preprocessor = DALIPreprocessor(train_files, train_idx_files, args.batch_size, num_preprocessing_threads, vector_size,
                                          dali_cpu=True, deterministic=False, training=True)
    val_preprocessor = DALIPreprocessor(val_files, val_idx_files, args.batch_size, num_preprocessing_threads, vector_size, dali_cpu=True,
                                        deterministic=False, training=False)

    train_input = train_preprocessor.get_device_dataset()
    val_input = val_preprocessor.get_device_dataset()

    # update epoch and learning rate if necessary
    epoch = args.epoch_to_resume + 1 if args.resume else 1
    args.init_lr = args.init_lr/(2*(epoch//args.lr_decay)) if args.resume else args.init_lr

    # define optimizer
    opt = tf.keras.optimizers.Adam(args.init_lr)
    opt = keras.mixed_precision.LossScaleOptimizer(opt)

    # define model
    if args.resume:
        # load model in SavedModel format
        #model = tf.keras.models.load_model(args.model)
        # load model saved with checkpoints
        model = DNA_net(args, vector_size, args.embedding_size, num_classes, vocab_size, args.dropout_rate) if args.DNA_model else AlexNet(args, vector_size, args.embedding_size, num_classes, vocab_size, args.dropout_rate)
        checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
        checkpoint.restore(os.path.join(args.ckpt, f'ckpt-{args.epoch_to_resume}')).expect_partial()

    else:
        model = DNA_net(args, vector_size, args.embedding_size, num_classes, vocab_size, args.dropout_rate) if args.DNA_model else AlexNet(args, vector_size, args.embedding_size, num_classes, vocab_size, args.dropout_rate)

    # define metrics
    loss = tf.losses.SparseCategoricalCrossentropy()
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    if hvd.rank() == 0:
        # create output directory
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

        # create directory for storing checkpoints
        ckpt_dir = os.path.join(args.output_dir, f'ckpts-rnd-{args.rnd}')
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)

        # create checkpoint object to save model
        checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)

        # create directory for storing logs
        tensorboard_dir = os.path.join(args.output_dir, 'logs')
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        writer = tf.summary.create_file_writer(tensorboard_dir)
        td_writer = open(os.path.join(args.output_dir, 'logs', f'training_data_rnd_{args.rnd}.tsv'), 'w')
        vd_writer = open(os.path.join(args.output_dir, 'logs', f'validation_data_rnd_{args.rnd}.tsv'), 'w')

        # create summary file
        with open(os.path.join(args.output_dir, f'training-summary-rnd-{args.rnd}.tsv'), 'w') as f:
            f.write(f'Date\t{datetime.datetime.now().strftime("%d/%m/%Y")}\nTime\t{datetime.datetime.now().strftime("%H:%M:%S")}\n'
                    f'Model\t{model}\nRound of training\t{args.rnd}\nNumber of classes\t{num_classes}\nEpochs\t{args.epochs}\n'
                    f'Vector size\t{vector_size}\nVocabulary size\t{vocab_size}\nEmbedding size\t{args.embedding_size}\n'
                    f'Dropout rate\t{args.dropout_rate}\nBatch size per gpu\t{args.batch_size}\n'
                    f'Global batch size\t{args.batch_size*hvd.size()}\nNumber of gpus\t{hvd.size()}\n'
                    f'Training set size\t{args.num_train_samples}\nValidation set size\t{args.num_val_samples}\n'
                    f'Number of steps per epoch\t{nstep_per_epoch}\nNumber of steps for validation dataset\t{val_steps}\n'
                    f'Initial learning rate\t{args.init_lr}\nLearning rate decay\t{args.lr_decay}')
            if args.DNA_model:
                f.write(f'n_rows\t{args.n_rows}\nn_cols\t{args.n_cols}\nkh_conv_1\t{args.kh_conv_1}\n'
                        f'kh_conv_2\t{args.kh_conv_2}\nkw_conv_1\t{args.kw_conv_1}*{args.embedding_size}\n'
                        f'kw_conv_2\t{args.kw_conv_2}\n')

    start = datetime.datetime.now()

    for batch, (reads, labels) in enumerate(train_input.take(nstep_per_epoch*args.epochs), 1):
        # get training loss
        loss_value, gradients = training_step(reads, labels, train_accuracy, loss, opt, model, batch == 1)

        if batch % 100 == 0 and hvd.rank() == 0:
            print(f'Epoch: {epoch} - Step: {batch} - learning rate: {opt.learning_rate.numpy()} - Training loss: {loss_value} - Training accuracy: {train_accuracy.result().numpy()*100}')
            # write metrics
            with writer.as_default():
                tf.summary.scalar("learning_rate", opt.learning_rate, step=batch)
                tf.summary.scalar("train_loss", loss_value, step=batch)
                tf.summary.scalar("train_accuracy", train_accuracy.result().numpy(), step=batch)
                writer.flush()
            td_writer.write(f'{epoch}\t{batch}\t{opt.learning_rate.numpy()}\t{loss_value}\t{train_accuracy.result().numpy()}\n')

        # evaluate model at the end of every epoch
        if batch % nstep_per_epoch == 0:
            for _, (reads, labels) in enumerate(val_input.take(val_steps)):
                testing_step(reads, labels, loss, val_loss, val_accuracy, model)

            # adjust learning rate
            if epoch % args.lr_decay == 0:
                current_lr = opt.learning_rate
                new_lr = current_lr / 2
                opt.learning_rate = new_lr

            if hvd.rank() == 0:
                print(f'Epoch: {epoch} - Step: {batch} - Validation loss: {val_loss.result().numpy()} - Validation accuracy: {val_accuracy.result().numpy()*100}')
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

            # define end of current epoch
            epoch += 1

    if hvd.rank() == 0:
        # save final embeddings
        emb_weights = model.get_layer('embedding').get_weights()[0]
        out_v = io.open(os.path.join(args.output_dir, f'embeddings_rnd_{args.rnd}.tsv'), 'w', encoding='utf-8')
        print(f'# embeddings: {len(emb_weights)}')
        for i in range(len(emb_weights)):
            vec = emb_weights[i]
            out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_v.close()

    end = datetime.datetime.now()

    if hvd.rank() == 0:
        total_time = end - start
        hours, seconds = divmod(total_time.seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        with open(os.path.join(args.output_dir, f'training-summary-rnd-{args.rnd}.tsv'), 'a') as f:
            f.write("\nTraining runtime:\t%02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))
        print("\nTraining runtime: %02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))
        td_writer.close()
        vd_writer.close()


if __name__ == "__main__":
    main()
