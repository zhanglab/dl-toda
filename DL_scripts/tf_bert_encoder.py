# https://keras.io/examples/nlp/masked_language_modeling/
import tensorflow as tf
import horovod.tensorflow as hvd
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.tfrecord as tfrec
import nvidia.dali.plugin.tf as dali_tf
from dataclasses import dataclass
import numpy as np
import datetime
import argparse
import glob
import math
import json
import sys
import os
import io

print(tf.__version__)
print(tf.__file__)

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


parser = argparse.ArgumentParser()
parser.add_argument('--tfrecords_dir', help="Path to directory containing tfrecords")
parser.add_argument('--output_dirname', help="Name of output directory")
parser.add_argument('--vocab_file', help="Path to the vocabulary file")
parser.add_argument('--k_value', default=12, type=int, help="Size of k-mers")
parser.add_argument('--epochs', default=1, type=int, help="Number of epochs")
parser.add_argument('--batch_size', default=16, type=int, help="batch size per gpu")
parser.add_argument('--learning_rate', default=0.0001, type=float, help="learning rate")
parser.add_argument('--sliding_window', default=1, type=int, help="Length of step when sliding window over read")
parser.add_argument('--classes_file', type=str, help='path to json file mapping species labels to rank labels')
parser.add_argument('--read_length', default=250, type=int, help="The length of simulated reads")
parser.add_argument('--model', type=str, help="path to pretrained model")
parser.add_argument('--ckpt', type=str, help="path to directory containing checkpoint files of pretrained model")
parser.add_argument('--epoch_to_resume', type=int, help="epoch to resume training of model")
parser.add_argument('--embed_dim', default=1280, type=int, help="dimension of kmers embeddings")
parser.add_argument('--num_heads', default=12, type=int, help="number of heads")
parser.add_argument('--num_layers', default=12, type=int, help="number of layers")
parser.add_argument('--ff_dim', default=768, type=int, help="dimension of feed-forward layer")
parser.add_argument('--buffer_size', default=10000, type=int, help="buffer size for shuffling reads")
args = parser.parse_args()
print(args)

# tfrecords_dir = sys.argv[1]
# vocab_file = sys.argv[2]
# classes_file = str(sys.argv[3]) # path to json file with list of species
# output_dirname = sys.argv[4]
# batch_size = int(sys.argv[5])
# k_value = int(sys.argv[6])
# read_length = int(sys.argv[7])
# sliding_window = int(sys.argv[8])
# mode = str(sys.argv[9])
# learning_rate = float(sys.argv[10])
# embed_dim = int(sys.argv[11])
# num_layers = int(sys.argv[12])
# num_heads = int(sys.argv[13])
# ff_dim = int(sys.argv[14])
# num_epochs = int(sys.argv[15])
# model = str(sys.argv[16])

# load vocabulary
with open(args.vocab_file, 'r') as f:
    vocab = f.readlines()

# load class_mapping file mapping label IDs to species
f = open(args.classes_file, 'r')
class_mapping = json.load(f)

# set-up configuration
@dataclass
class Config:
    # MODE = mode
    # K_VALUE = k_value
    # READ_LENGTH = read_length
    MAX_LENGTH = args.read_length - args.k_value + 1 if args.sliding_window == 1 else args.read_length // args.k_value
    # BATCH_SIZE = batch_size
    # LR = 0.0001
    # LR = learning_rate
    VOCAB_SIZE = len(vocab) + 1 # include token for padding --> 0
    # EMBED_DIM = 1280
    # NUM_HEAD = 12  # used in bert model
    # FF_DIM = 768  # used in bert model --> hidden size
    # NUM_LAYERS = 12 # used in bert model
    # BUFFER_SIZE = 10000
    # EPOCHS = 1
    # MODEL = model
    NUM_CLASSES = len(class_mapping)

    TRAIN_ACCURACY = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    VAL_ACCURACY = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    TRAIN_LOSS = tf.keras.metrics.Mean(name="train_loss")
    VAL_LOSS = tf.keras.metrics.Mean(name="val_loss")

    OPT = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(learning_rate=args.learning_rate))

    WAIT = 0
    FOUND_MIN = False
    BEST = np.Inf
    VAL_LOSS_BEFORE = -1
    LOWEST_VAL_LOSS = 1
    BEST_WEIGHTS = None
    MIN_EPOCH = 0
    PATIENCE = 0
    STOP_TRAINING = False
    BEST = np.Inf


config = Config()


def on_epoch_end(epoch, num_train_batches, model):
    # Early stopping
    val_loss = config.VAL_LOSS.result()
    if config.WAIT < 2:
        model_checkpoint(val_loss, epoch, model)
    else:
        config.FOUND_MIN = True
        early_stopping(val_loss)

def model_checkpoint(val_loss, epoch, model):
        # ModelCheckpoint
        if val_loss < config.BEST:
            config.BEST = val_loss
            config.LOWEST_VAL_LOSS = val_loss
            config.BEST_WEIGHTS = model.get_weights()
            config.WAIT = 0
            config.MIN_EPOCH = epoch
        else:
            config.WAIT += 1

def early_stopping(val_loss):
    # Calculate percent difference
    if abs(100 * (val_loss - config.VAL_LOSS_BEFORE) / config.VAL_LOSS_BEFORE) < 5:
        config.PATIENCE += 1
        if config.PATIENCE == 2:
            if config.OPT.learning_rate == 0.0001:
                config.OPT.learning_rate = 0.00001
                config.PATIENCE = 0
            else:
                config.STOP_TRAINING = True
    else:
        config.PATIENCE = 0

    config.VAL_LOSS_BEFORE = val_loss



def count_num_reads(tfrecords_dir, data):
    in_files = glob.glob(os.path.join(tfrecords_dir, f"{data}*-read_count"))
    n_reads = 0
    for in_f in in_files:
        with open(in_f, 'r') as f:
            for line in f:
                n_reads += int(line.rstrip())
    return n_reads


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
                                     "read_masked": tfrec.VarLenFeature([], tfrec.int64, 0),
                                     "weights": tfrec.VarLenFeature([], tfrec.float32, 0.0),
                                     "label": tfrec.FixedLenFeature([1], tfrec.int64, -1)})
    # retrieve reads and labels and copy them to the gpus
    reads = inputs["read"].gpu()
    reads_masked = inputs["read_masked"].gpu()
    weights= inputs["weights"].gpu()
    labels = inputs["label"].gpu()
    return (reads, reads_masked, weights, labels)

class DALIPreprocessor(object):
    def __init__(self, filenames, idx_filenames, batch_size, vector_size, initial_fill,
               deterministic=False, training=False):

        device_id = hvd.local_rank()
        shard_id = hvd.rank()
        num_gpus = hvd.size()
        self.pipe = get_dali_pipeline(tfrec_filenames=filenames, tfrec_idx_filenames=idx_filenames, batch_size=batch_size,
                                      device_id=device_id, shard_id=shard_id, initial_fill=initial_fill, num_gpus=num_gpus,
                                      training=training, seed=7 * (1 + hvd.rank()) if deterministic else None)

        self.daliop = dali_tf.DALIIterator()

        self.batch_size = batch_size
        self.device_id = device_id

        self.dalidataset = dali_tf.DALIDataset(fail_on_device_mismatch=False, pipeline=self.pipe,
            # output_shapes=((batch_size, vector_size), (batch_size)),            
            output_shapes=((batch_size, vector_size), (batch_size, vector_size), (batch_size, vector_size), (batch_size)),
            batch_size=batch_size, output_dtypes=(tf.int64, tf.int64, tf.float32, tf.int64), device_id=device_id)
 
    def get_device_dataset(self):
        return self.dalidataset


def load_dataset(tfrecords_dir, set_type):
    # Get list of TFRecord files
    tfrecords = glob.glob(os.path.join(tfrecords_dir, f"{set_type}*.tfrec"))
    print(tfrecords)
    num_parallel_reads = os.cpu_count() if len(tfrecords) > os.cpu_count() else len(tfrecords)
    dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=num_parallel_reads)
    dataset = dataset.map(map_func=decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(args.buffer_size)
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def decode_fn(proto_example):
    features = {
        "read": tf.io.FixedLenFeature([config.MAX_LENGTH], dtype=tf.int64, default_value=[0]*config.MAX_LENGTH),
        "read_masked": tf.io.FixedLenFeature([config.MAX_LENGTH], dtype=tf.int64, default_value=[0]*config.MAX_LENGTH),
        "weights": tf.io.FixedLenFeature([config.MAX_LENGTH], dtype=tf.float32, default_value=[0]*config.MAX_LENGTH),
        "label": tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
    }
    # load one example
    parsed_example = tf.io.parse_single_example(serialized=proto_example, features=features)
    read = parsed_example['read']
    read_masked = parsed_example['read_masked']
    weights = parsed_example['weights']
    labels = parsed_example['label']
    return read, read_masked, weights, labels


def bert_module(query, key, value, i):
    # Multi headed self-attention
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=args.num_heads,
        key_dim=args.embed_dim // args.num_heads,
        name="encoder_{}/multiheadattention".format(i),
    )(query, key, value)
    attention_output = tf.keras.layers.Dropout(0.1, name="encoder_{}/att_dropout".format(i))(
        attention_output
    )
    attention_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/att_layernormalization".format(i)
    )(query + attention_output)

    # Feed-forward layer
    ffn = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(args.ff_dim, activation="relu"),
            tf.keras.layers.Dense(args.embed_dim),
        ],
        name="encoder_{}/ffn".format(i),
    )
    ffn_output = ffn(attention_output)
    ffn_output = tf.keras.layers.Dropout(0.1, name="encoder_{}/ffn_dropout".format(i))(
        ffn_output
    )
    sequence_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/ffn_layernormalization".format(i)
    )(attention_output + ffn_output)
    return sequence_output


def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


@tf.function
# def training_step(inputs, train_accuracy, loss, training_loss, opt, model):
def training_step(inputs, loss, model):

    reads, reads_masked, sample_weight, labels = inputs

    with tf.GradientTape() as tape:
        probs = model(reads_masked, training=True)
        # get the loss
        loss_value = loss(reads, probs, sample_weight=sample_weight)
        # scale the loss (multiply the loss by a factor) to avoid numeric underflow
        scaled_loss = config.OPT.get_scaled_loss(loss_value)

    # get the scaled gradients
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    # get the unscaled gradients
    grads = config.OPT.get_unscaled_gradients(scaled_gradients)
    # Update weights
    config.OPT.apply_gradients(zip(grads, model.trainable_variables))

    # update training accuracy and
    config.TRAIN_ACCURACY.update_state(reads, probs, sample_weight=sample_weight)
    config.TRAIN_LOSS.update_state(loss_value, sample_weight=sample_weight)

    # return loss_value, grads, probs, scaled_loss, reads, reads_masked, sample_weight
    return reads, labels, sample_weight, probs


@tf.function
# def testing_step(inputs, loss, val_loss, val_accuracy, model):
def testing_step(inputs, loss, model):
    reads, reads_masked, sample_weight, labels = inputs
    probs = model(reads_masked, training=False)
    config.VAL_ACCURACY.update_state(reads, probs, sample_weight=sample_weight)
    loss_value = loss(reads, probs, sample_weight=sample_weight)
    config.VAL_LOSS.update_state(loss_value, sample_weight=sample_weight)


def create_masked_language_bert_model():
    inputs = tf.keras.layers.Input((config.MAX_LENGTH,), dtype=tf.int64)

    word_embeddings = tf.keras.layers.Embedding(
        config.VOCAB_SIZE, args.embed_dim, name="word_embedding"
    )(inputs)
    position_embeddings = tf.keras.layers.Embedding(
        input_dim=config.MAX_LENGTH,
        output_dim=args.embed_dim,
        weights=[get_pos_encoding_matrix(config.MAX_LENGTH, args.embed_dim)],
        name="position_embedding",
    )(tf.range(start=0, limit=config.MAX_LENGTH, delta=1))
    embeddings = word_embeddings + position_embeddings
    embeddings = tf.keras.layers.LayerNormalization()(embeddings)

    encoder_output = embeddings
    for i in range(args.num_layers):
        encoder_output = bert_module(encoder_output, encoder_output, encoder_output, i)

    output = tf.keras.layers.Dense(config.VOCAB_SIZE, name="mlm_cls", activation="softmax")(
        encoder_output
    )

    model = tf.keras.models.Model(inputs, output, name='BERT')

    return model


def create_read_classifier(pretrained_bert_model):
    inputs = tf.keras.layers.Input((config.MAX_LENGTH,), dtype=tf.int64)
    x = pretrained_bert_model(inputs)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(1000, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
    x = tf.keras.layers.Dense(config.NUM_CLASSES)(x)
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
    outputs = tf.keras.layers.Activation('softmax', dtype='float32',)(x)
    classifer_model = tf.keras.models.Model(inputs, outputs, name="classification")
    
    return classifer_model


# # load training and validation datasets
# train_dataset = load_dataset(args.tfrecords_dir, 'train')
# val_dataset = load_dataset(args.tfrecords_dir, 'val')


# for inputs in train_dataset.take(1):
#     print(inputs)

# Get training and validation tfrecords
train_files = sorted(glob.glob(os.path.join(args.tfrecords_dir, 'train*.tfrec')))
train_idx_files = sorted(glob.glob(os.path.join(args.tfrecords_dir, 'idx_files', 'train*.idx')))
val_files = sorted(glob.glob(os.path.join(args.tfrecords_dir, 'val*.tfrec')))
val_idx_files = sorted(glob.glob(os.path.join(args.tfrecords_dir, 'idx_files', 'val*.idx')))

# load training and validation datasets
train_preprocessor = DALIPreprocessor(train_files, train_idx_files, args.batch_size, config.MAX_LENGTH, args.buffer_size,
                                           deterministic=False, training=True)
val_preprocessor = DALIPreprocessor(val_files, val_idx_files, args.batch_size, config.MAX_LENGTH, args.buffer_size,
                                    deterministic=False, training=True)
train_input = train_preprocessor.get_device_dataset()
val_input = val_preprocessor.get_device_dataset()

# create dtype policy
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

# compute number of steps/batches per epoch
n_train_reads = count_num_reads(args.tfrecords_dir, 'train')
n_val_reads = count_num_reads(args.tfrecords_dir, 'val')
# n_train_reads = 100000
# n_val_reads = 50000
nstep_per_epoch = n_train_reads // args.batch_size
val_steps = n_val_reads // args.batch_size

print(f'# train reads: {n_train_reads} - # steps: {nstep_per_epoch}')
print(f'# val reads: {n_val_reads} - # steps: {val_steps}')

# Define loss and create model
loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
model = create_masked_language_bert_model()
model.summary()

# define optimizer
# opt = tf.keras.optimizers.Adam(learning_rate=config.LR)
# opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

if hvd.rank() == 0:
    # create directory for storing checkpoints
    ckpt_dir = os.path.join(args.output_dirname, f'ckpts')
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    # create checkpoint object to save model
    checkpoint = tf.train.Checkpoint(model=model, optimizer=config.OPT)

    # create directory for storing logs
    tensorboard_dir = os.path.join(args.output_dirname, f'logs')
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    writer = tf.summary.create_file_writer(tensorboard_dir)
    td_writer = open(os.path.join(tensorboard_dir, f'training_data.tsv'), 'w')
    vd_writer = open(os.path.join(tensorboard_dir, f'validation_data.tsv'), 'w')

# for batch, inputs in enumerate(train_input.take(nstep_per_epoch * config.EPOCHS), 1):
#     reads, reads_masked, sample_weight, labels = inputs
#     print(reads)
#     print(reads_masked)
#     print(sample_weight)
#     print(labels)

# start training
start = datetime.datetime.now()
epoch = 1
for batch, inputs in enumerate(train_input.take(nstep_per_epoch * args.epochs), 1):
    reads, labels, sample_weight, probs = training_step(inputs, loss, model)
    # get training loss
    # loss_value, gradients = training_step(inputs, train_accuracy, loss, training_loss, opt, bert_masked_model)
    # loss_value, gradients, probs, scaled_loss, reads, labels, sample_weight = training_step(inputs, loss, bert_masked_model)
    # print(f"reads: {reads[0]} - {len(reads[0])}")
    # print(f"labels: {labels}")
    # print(f"sample_weight: {sample_weight}")
    # print(f"probs: {probs}")
    # print(f"loss_value: {loss_value}")
    # print(f"scaled_loss: {scaled_loss}")
    # np.save(os.path.join(output_dirname, 'reads.npy'), reads)
    # np.save(os.path.join(output_dirname, 'labels.npy'), labels)
    # np.save(os.path.join(output_dirname, 'sample_weight.npy'), sample_weight)
    # np.save(os.path.join(output_dirname, 'probs.npy'), probs)
    # break
    if batch % 100 == 0:
        if hvd.rank() == 0:
            print(f'Epoch: {epoch} - Step: {batch} - learning rate: {config.OPT.learning_rate.numpy()} - Training loss: {config.TRAIN_LOSS.result().numpy()} - Training accuracy: {config.TRAIN_ACCURACY.result().numpy() * 100}')
        
            # write training metrics
            with writer.as_default():
                tf.summary.scalar("learning_rate", config.OPT.learning_rate, step=batch)
                tf.summary.scalar("train_loss", config.TRAIN_LOSS.result().numpy(), step=batch)
                tf.summary.scalar("train_accuracy", config.TRAIN_ACCURACY.result().numpy(), step=batch)
                writer.flush()
            td_writer.write(f'{epoch}\t{batch}\t{config.OPT.learning_rate.numpy()}\t{config.TRAIN_LOSS.result().numpy()}\t{config.TRAIN_ACCURACY.result().numpy()}\n')
    
    if math.isnan(config.TRAIN_LOSS.result().numpy()):
        print(f'Epoch: {epoch} - Step: {batch} - learning rate: {config.OPT.learning_rate.numpy()} - Training loss: {config.TRAIN_LOSS.result().numpy()} - Training accuracy: {config.TRAIN_ACCURACY.result().numpy() * 100}')
        # np.save(os.path.join(output_dirname, f'reads-{hvd.rank()}.npy'), reads)
        # np.save(os.path.join(output_dirname, f'labels-{hvd.rank()}.npy'), labels)
        # np.save(os.path.join(output_dirname, f'sample_weight_{hvd.rank()}.npy'), sample_weight)
        # np.save(os.path.join(output_dirname, f'probs-{hvd.rank()}.npy'), probs)
        break

    # evaluate and save model at the end of every epoch
    if batch % nstep_per_epoch == 0:
        for _, inputs in enumerate(val_input.take(val_steps)):
            # testing_step(inputs, loss, val_loss, val_accuracy, bert_masked_model)
            testing_step(inputs, loss, model)

        if hvd.rank() == 0:
            print(f'Epoch: {epoch} - Step: {batch} - Validation loss: {config.VAL_LOSS.result().numpy()} - Validation accuracy: {config.VAL_ACCURACY.result().numpy()*100}')
            
            # save weights
            checkpoint.save(os.path.join(ckpt_dir, 'ckpt'))
            model.save(os.path.join(args.output_dirname, f'model-{epoch}'))

            # write validation metrics
            with writer.as_default():
                tf.summary.scalar("val_loss", config.VAL_LOSS.result().numpy(), step=epoch)
                tf.summary.scalar("val_accuracy", config.VAL_ACCURACY.result().numpy(), step=epoch)
                writer.flush()
            vd_writer.write(f'{epoch}\t{batch}\t{config.VAL_LOSS.result().numpy()}\t{config.VAL_ACCURACY.result().numpy()}\n')

        # compare validation losses
        on_epoch_end(epoch, batch, model)

        if config.STOP_TRAINING:
            model.set_weights(config.BEST_WEIGHTS)
            checkpoint.save(os.path.join(ckpt_dir, 'ckpt-minloss'))
            model.save(os.path.join(args.output_dirname, f'model-minloss'))

        epoch += 1

end = datetime.datetime.now()

if hvd.rank() == 0:
    # save final embeddings
    emb_weights = model.get_layer('word_embedding').get_weights()[0]
    out_v = io.open(os.path.join(args.output_dirname, f'word_embeddings.tsv'), 'w', encoding='utf-8')
    print(f'# embeddings: {len(emb_weights)}')
    for i in range(len(emb_weights)):
        vec = emb_weights[i]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_v.close()


if hvd.rank() == 0:
    # write report
    total_time = end - start
    days, seconds = divmod(total_time.seconds, 86400)
    hours, seconds = divmod(total_time.seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    with open(os.path.join(args.output_dirname, f'training-summary.tsv'), 'a') as f:
        for k, v in vars(args).items():
            f.write(f"{k}:\t{v}\n")
        f.write(f'#GPUs:\t{len(gpus)}\n')
        f.write("\nTraining runtime:\t%02d:%02d:%02d:%02d.%d\n" % (days, hours, minutes, seconds, total_time.microseconds))
    td_writer.close()
    vd_writer.close()

