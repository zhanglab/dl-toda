# code from https://www.tensorflow.org/text/tutorials/transformer
import logging
import time
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
#import tensorflow_text
import horovod.tensorflow as hvd
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.tfrecord as tfrec
import nvidia.dali.plugin.tf as dali_tf
import sys
import glob
import os
import json

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
            output_shapes=((batch_size, vector_size), (batch_size)),
            batch_size=batch_size, output_dtypes=(tf.int64, tf.int64), device_id=device_id)

    def get_device_dataset(self):
        return self.dalidataset

# MAX_TOKENS = 128
# BUFFER_SIZE = 20000
# BATCH_SIZE = 64
# def prepare_batch(pt, en):
#     pt = tokenizers.pt.tokenize(pt)      # Output is ragged.
#     pt = pt[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
#     pt = pt.to_tensor()  # Convert to 0-padded dense Tensor
#
#     en = tokenizers.en.tokenize(en)
#     en = en[:, :(MAX_TOKENS+1)]
#     en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
#     en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens
#
#     return (pt, en_inputs), en_labels


# def make_batches(ds):
#     return (
#       ds
#       .shuffle(BUFFER_SIZE)
#       .batch(BATCH_SIZE)
#       .map(prepare_batch, tf.data.AUTOTUNE)
#       .prefetch(buffer_size=tf.data.AUTOTUNE))


def positional_encoding(length, depth):
    """
    function that calculates the positional encoding of tokens
    that is added to the embedding vectors

    """
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)
    # concatenate the vectors of sines and cosines
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)],axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    """ Class that creates a positional embedding layer that looks-up
    a token's embedding vector and adds the position vector
    example: embed_en = PositionalEmbedding(vocab_size=tokenizers.en.get_vocab_size(), d_model=512)
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class BaseAttention(tf.keras.layers.Layer):
    """ Implementation of the attention layer"""
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    """ implementation of the cross-attention layer that connects
        the encoder and decoder.
    """
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    """ implementation of the global self attention layer
        responsible for processing the context sequence, and propagating information along its length
    """
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    """ implementation of the causal self attention layer"""
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
          tf.keras.layers.Dense(dff, activation='relu'),
          tf.keras.layers.Dense(d_model),
          tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
          x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
               dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                 d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x


class Transformer(tf.keras.Model):
    """ example: transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    dropout_rate=dropout_rate)"""
    def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context, x = inputs

        context = self.encoder(context)  # (batch_size, context_len, d_model)

        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass

        # Return the final output and the attention weights.
        return logits


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def masked_loss(label, pred):
    """ function to apply a padding mask when calculating the loss"""
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)


def load_tfrecords(proto_example):
    print('in read_tfrecord', tf.executing_eagerly())
    data_description = {
        'read': tf.io.VarLenFeature(tf.int64),
        'label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
    }
    # load one example
    parsed_example = tf.io.parse_single_example(serialized=proto_example, features=data_description)
    read = parsed_example['read']
    label = tf.cast(parsed_example['label'], tf.int64)
    print(read, label)
    # read = tf.sparse.to_dense(read)
    return read, label


def make_batches(tfrecord_path, batch_size, vector_size, num_classes):
    """ Return reads and labels """
    # Load data as shards
    dataset = tf.data.Dataset.list_files(tfrecord_path)
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x), num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                 deterministic=False)
    dataset = dataset.map(map_func=load_tfrecords, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=(tf.TensorShape([vector_size]), tf.TensorShape([num_classes])),)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    tfrec_dir = str(sys.argv[1])
    vocab = str(sys.argv[2])
    class_mapping = str(sys.argv[3])
    train_reads_per_epoch = int(sys.argv[4])
    val_reads_per_epoch = int(sys.argv[5])

    with open(class_mapping, 'r') as f:
        classes_dict = json.load(f)
        num_classes = len(classes_dict)

    with open(vocab, 'r') as f:
        content = f.readlines()
        vocab_size = len(content)


    batch_size = 64
    max_read_size = 250
    k_value = 4
    initial_fill = 1000000
    vector_size = (max_read_size - k_value + 1) * 2

    epochs = 1
    nstep_per_epoch = train_reads_per_epoch // (batch_size * hvd.size())
    val_steps = val_reads_per_epoch // (batch_size * hvd.size())

    train_files = sorted(glob.glob(os.path.join(tfrec_dir, 'train*.tfrec')))
    train_idx_files = sorted(glob.glob(os.path.join(tfrec_dir, 'idx_files', 'train*.idx')))
    val_files = sorted(glob.glob(os.path.join(tfrec_dir, 'val*.tfrec')))
    val_idx_files = sorted(glob.glob(os.path.join(tfrec_dir, 'idx_files', 'val*.idx')))

    train_preprocessor = DALIPreprocessor(train_files, train_idx_files, batch_size, vector_size, initial_fill,
                                          deterministic=False, training=True)
    val_preprocessor = DALIPreprocessor(val_files, val_idx_files, batch_size, vector_size, initial_fill,
                                        deterministic=False, training=True)

    train_input = train_preprocessor.get_device_dataset()
    val_input = val_preprocessor.get_device_dataset()

    # make training batches
    train_batches = train_input.take(nstep_per_epoch*epochs)
    print(train_batches)
    # make validation batches
    val_batches = val_input.take(val_steps)
    print(val_batches)
    # for batch, (reads, labels) in enumerate(train_input.take(nstep_per_epoch*epochs), 1):
    # for batch, (reads, labels) in enumerate(train_input.take(1), 1):
    #     print(reads, labels)
        # break

    # train_batches = make_batches(tfrec_dir, batch_size, vector_size, num_classes)
    #
    # for (read, label) in train_batches.take(1):
    #     break
    #
    # print(read.shape)
    # print(label.shape)

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    # instantiate the transformer model
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        dropout_rate=dropout_rate)

    # output = transformer((reads, reads))
    #
    # print(reads.shape)
    # print(output.shape)
    #
    attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
    # print(attn_scores.shape)  # (batch, heads, target_seq, input_seq)
    # print(transformer.summary())
    #
    # start training
    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    transformer.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy])

    transformer.fit(train_batches,
                    epochs=1,
                    validation_data=val_batches)

    # download the dataset
    # examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
    #                                with_info=True,
    #                                as_supervised=True)
    #
    # train_examples, val_examples = examples['train'], examples['validation']
    #
    # for pt_examples, en_examples in train_examples.batch(3).take(1):
    #     print('> Examples in Portuguese:')
    #     for pt in pt_examples.numpy():
    #         print(pt.decode('utf-8'))
    #     print()
    #
    #     print('> Examples in English:')
    #     for en in en_examples.numpy():
    #         print(en.decode('utf-8'))
    #
    # # tokenize the text
    # model_name = 'ted_hrlr_translate_pt_en_converter'
    # tf.keras.utils.get_file(
    #     f'{model_name}.zip',
    #     f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
    #     cache_dir='.', cache_subdir='', extract=True
    # )
    #
    # # load tokenizers
    # tokenizers = tf.saved_model.load(model_name)
    #
    # print([item for item in dir(tokenizers.en) if not item.startswith('_')])
    #
    # # use tokenize method to convert a batch of strings to a padded-batch of token IDs
    # encoded = tokenizers.en.tokenize(en_examples)
    #
    # print('> This is a padded-batch of token IDs:')
    # for row in encoded.to_list():
    #     print(row)
    #
    # # use the detokenize method to convert the token IDs back to human-readable text
    # round_trip = tokenizers.en.detokenize(encoded)
    #
    # print('> This is human-readable text:')
    # for line in round_trip.numpy():
    #     print(line.decode('utf-8'))
    #
    # # Create training and validation set batches.
    # train_batches = make_batches(train_examples)
    # val_batches = make_batches(val_examples)
    #
    # for (pt, en), en_labels in train_batches.take(1):
    #     break
    # print('CHECK BATCHES')
    # print(pt.shape)
    # print(en.shape)
    # print(en_labels.shape)
    #
    # print('positional embedding layer')
    # embed_pt = PositionalEmbedding(vocab_size=tokenizers.pt.get_vocab_size(), d_model=512)
    # embed_en = PositionalEmbedding(vocab_size=tokenizers.en.get_vocab_size(), d_model=512)
    #
    # pt_emb = embed_pt(pt)
    # en_emb = embed_en(en)
    #
    # print(en_emb._keras_mask)
    #
    # print('cross attention layer')
    # sample_ca = CrossAttention(num_heads=2, key_dim=512)
    #
    # print(pt_emb.shape)
    # print(en_emb.shape)
    # print(sample_ca(en_emb, pt_emb).shape)
    #
    # sample_gsa = GlobalSelfAttention(num_heads=2, key_dim=512)
    #
    # print(pt_emb.shape)
    # print(sample_gsa(pt_emb).shape)
    #
    # sample_csa = CausalSelfAttention(num_heads=2, key_dim=512)
    #
    # print(en_emb.shape)
    # print(sample_csa(en_emb).shape)
    #
    # out1 = sample_csa(embed_en(en[:, :3]))
    # out2 = sample_csa(embed_en(en))[:, :3]
    #
    # print(tf.reduce_max(abs(out1 - out2)).numpy())
    #
    # sample_ffn = FeedForward(512, 2048)
    #
    # print(en_emb.shape)
    # print(sample_ffn(en_emb).shape)
    #
    # print('TEST ENCODER LAYER')
    # sample_encoder_layer = EncoderLayer(d_model=512, num_heads=8, dff=2048)
    #
    # print(pt_emb.shape)
    # print(sample_encoder_layer(pt_emb).shape)
    #
    # print('TEST ENCODER')
    # # Instantiate the encoder.
    # sample_encoder = Encoder(num_layers=4,
    #                          d_model=512,
    #                          num_heads=8,
    #                          dff=2048,
    #                          vocab_size=8500)
    #
    # sample_encoder_output = sample_encoder(pt, training=False)
    #
    # # Print the shape.
    # print(pt.shape)
    # print(sample_encoder_output.shape)  # Shape `(batch_size, input_seq_len, d_model)`.
    #
    # print('TEST DECODER LAYER')
    # sample_decoder_layer = DecoderLayer(d_model=512, num_heads=8, dff=2048)
    #
    # sample_decoder_layer_output = sample_decoder_layer(
    #     x=en_emb, context=pt_emb)
    #
    # print(en_emb.shape)
    # print(pt_emb.shape)
    # print(sample_decoder_layer_output.shape)  # `(batch_size, seq_len, d_model)`
    #
    # print('TEST DECODER')
    # # Instantiate the decoder.
    # sample_decoder = Decoder(num_layers=4,
    #                          d_model=512,
    #                          num_heads=8,
    #                          dff=2048,
    #                          vocab_size=8000)
    #
    # output = sample_decoder(
    #     x=en,
    #     context=pt_emb)
    #
    # # Print the shapes.
    # print(en.shape)
    # print(pt_emb.shape)
    # print(output.shape)
    #
    # print(sample_decoder.last_attn_scores.shape)  # (batch, heads, target_seq, input_seq))
    #
    # print('TEST TRANSFORMER')
    #
    # num_layers = 4
    # d_model = 128
    # dff = 512
    # num_heads = 8
    # dropout_rate = 0.1
    #
    # # instantiate the transformer model
    # transformer = Transformer(
    #     num_layers=num_layers,
    #     d_model=d_model,
    #     num_heads=num_heads,
    #     dff=dff,
    #     input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    #     target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    #     dropout_rate=dropout_rate)
    #
    # output = transformer((pt, en))
    #
    # print(en.shape)
    # print(pt.shape)
    # print(output.shape)

# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, d_model, warmup_steps=4000):
#         super().__init__()
#
#         self.d_model = d_model
#         self.d_model = tf.cast(self.d_model, tf.float32)
#
#         self.warmup_steps = warmup_steps
#
#     def __call__(self, step):
#         step = tf.cast(step, dtype=tf.float32)
#         arg1 = tf.math.rsqrt(step)
#         arg2 = step * (self.warmup_steps ** -1.5)
#
#         return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
#
# learning_rate = CustomSchedule(d_model)
#
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
#                                      epsilon=1e-9)