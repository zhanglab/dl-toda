from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import glob
import json
import math
import glob
import re
import os
import sys
import datetime
import numpy as np
import six
import tensorflow as tf
import tensorflow_models as tfm
import horovod.tensorflow as hvd
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.tfrecord as tfrec
import nvidia.dali.plugin.tf as dali_tf


# set seed
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)
tf.experimental.numpy.random.seed(seed)

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


class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               seq_length,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    """Constructs BertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range
    # add seq_length
    self.seq_length = seq_length

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None,seq_length=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.io.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def PositionalEncoding(config, seq_length, width):
    assert_op = tf.debugging.assert_less_equal(seq_length, config.max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      # full_position_embeddings = tf.compat.v1.get_variable(
      #     name="position_embeddings",
      #     shape=[config.max_position_embeddings, config.hidden_size],
      #     initializer=create_initializer(config.initializer_range))

      weights_initializer = create_initializer(config.initializer_range)
      full_position_embeddings = tf.Variable(
          initial_value=weights_initializer(shape=[config.max_position_embeddings, config.hidden_size]),
          name="position_embeddings")
      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                       [seq_length, -1])

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = []
      num_dims = 3
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      return position_embeddings


class TokenTypeEncoding(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.seq_length = config.seq_length
        self.width = config.hidden_size
        self.token_type_vocab_size = config.type_vocab_size
        self.weights_initializer = create_initializer(config.initializer_range)
        self.token_type_table = tf.Variable(
            initial_value=self.weights_initializer(shape=[self.token_type_vocab_size, self.width]),
            name="token_type_embeddings")

    def __call__(self, token_type_ids):
        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, self.token_type_table)

        return token_type_embeddings


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
    # return tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
    A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.

      Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    # batch_size = tf.shape(from_tensor)[0]
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    # from_seq_length = tf.shape(from_tensor)[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]
    # to_seq_length = tf.shape(from_tensor)[1]

    to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

    Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
    """
    # if name is None:
    #     name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

      Args:
        tensor: A tf.Tensor to check the rank of.
        expected_rank: Python integer or list of integers, expected rank.
        name: Optional name of the tensor for the error message.

      Raises:
        ValueError: If the expected shape doesn't match the actual shape.
      """
    # if name is None:
    #     name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor
    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
    input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.initializer_range = config.initializer_range
        self.size_per_head = int(config.hidden_size / config.num_attention_heads)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        # `query_layer` = [B*F, N*H]
        self.query_layer = tf.keras.layers.Dense(self.num_attention_heads * self.size_per_head,
            name='query', kernel_initializer=create_initializer(self.initializer_range))
        # `key_layer` = [B*T, N*H]
        self.key_layer = tf.keras.layers.Dense(self.num_attention_heads * self.size_per_head,
            name='key', kernel_initializer=create_initializer(self.initializer_range))
        # `value_layer` = [B*T, N*H]
        self.value_layer = tf.keras.layers.Dense(self.num_attention_heads * self.size_per_head,
            name='value', kernel_initializer=create_initializer(self.initializer_range))


    def __call__(self, from_tensor, to_tensor, attention_mask, do_return_2d_tensor):
        batch_size = tf.shape(from_tensor)[0]
        from_seq_length = tf.shape(from_tensor)[1]
        to_seq_length = tf.shape(to_tensor)[1]
        
        from_tensor_2d = reshape_to_matrix(from_tensor)
        to_tensor_2d = reshape_to_matrix(to_tensor)
    
        # `query_layer` = [B, N, F, H]
        query_layer = self.query_layer(from_tensor_2d)
        # print(f'query_layer: {query_layer}\t shape: {tf.shape(query_layer)}')
        
        # `key_layer` = [B, N, T, H]
        key_layer = self.key_layer(to_tensor_2d)
        
        # `value_layer` = [B*T, N*H]
        value_layer = self.value_layer(to_tensor_2d)
        

        # `query_layer` = [B, N, F, H]
        query_layer = transpose_for_scores(query_layer, batch_size,
                                         self.num_attention_heads, from_seq_length,
                                         self.size_per_head)

        # `key_layer` = [B, N, T, H]
        key_layer = transpose_for_scores(key_layer, batch_size, self.num_attention_heads,
                                       to_seq_length, self.size_per_head)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores,
                                     1.0 / math.sqrt(float(self.size_per_head)))

        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = dropout(attention_probs, self.attention_probs_dropout_prob)

        # `value_layer` = [B, T, N, H]
        value_layer = tf.reshape(
            value_layer,
            [batch_size, to_seq_length, self.num_attention_heads, self.size_per_head])

        # `value_layer` = [B, N, T, H]
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

        # `context_layer` = [B, N, F, H]
        context_layer = tf.matmul(attention_probs, value_layer)

        # `context_layer` = [B, F, N, H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

        if do_return_2d_tensor:
          # `context_layer` = [B*F, N*H]
          context_layer = tf.reshape(
              context_layer,
              [batch_size * from_seq_length, self.num_attention_heads * self.size_per_head])
        else:
          # `context_layer` = [B, F, N*H]
          context_layer = tf.reshape(
              context_layer,
              [batch_size, from_seq_length, self.num_attention_heads * self.size_per_head])

        return context_layer



class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.attention_layer = AttentionLayer(config=config)
        self.attention_output = tf.keras.layers.Dense(self.hidden_size)
        self.intermediate_layer = tf.keras.layers.Dense(self.intermediate_size, activation="gelu")
        self.layer_output = tf.keras.layers.Dense(self.hidden_size)
        self.layer_norm = tf.keras.layers.LayerNormalization()


    def __call__(self, input_tensor, attention_mask, do_return_2d_tensor, do_return_all_layers):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
              "The hidden size (%d) is not a multiple of the number of attention "
              "heads (%d)" % (self.hidden_size, self.num_attention_heads))

        attention_head_size = int(self.hidden_size / self.num_attention_heads)
        input_shape = get_shape_list(input_tensor, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        input_width = input_shape[2]

        # The Transformer performs sum residuals on all layers so the input needs
        # to be the same as the hidden size.
        if input_width != self.hidden_size:
            raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, self.hidden_size))


        # We keep the representation as a 2D tensor to avoid re-shaping it back and
        # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
        # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
        # help the optimizer.
        prev_output = reshape_to_matrix(input_tensor)

        all_layer_outputs = []
        for layer_idx in range(self.num_hidden_layers):
            layer_input = prev_output
            attention_heads = []
            attention_head = self.attention_layer(input_tensor, input_tensor, attention_mask, do_return_2d_tensor)
            attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
            attention_output = attention_heads[0]
        else:
            # In the case where we have other sequences, we just concatenate
            # them to the self-attention head before the projection.
            attention_output = tf.concat(attention_heads, axis=-1)

        attention_output = self.attention_output(attention_output)
        attention_output = dropout(attention_output, self.hidden_dropout_prob)
        attention_output = self.layer_norm(attention_output)

        intermediate_output = self.intermediate_layer(attention_output)

        layer_output = self.layer_output(intermediate_output)
        layer_output = dropout(layer_output, self.hidden_dropout_prob)
        layer_output = self.layer_norm(layer_output)

        prev_output = layer_output

        all_layer_outputs.append(layer_output)

        if do_return_all_layers:
            final_outputs = []
            for layer_output in all_layer_outputs:
                final_output = reshape_from_matrix(layer_output, input_shape)
                final_outputs.append(final_output)
            return final_outputs
        else:
            final_output = reshape_from_matrix(prev_output, input_shape)
            return final_output


class BertModel(tf.keras.Model):

    def __init__(self, config, is_training, *args, **kwargs):
        super().__init__()
        self.seq_length = config.seq_length
        self.width = config.hidden_size
        self.dropout_prob = config.hidden_dropout_prob
        self.num_layers = config.num_hidden_layers
        
        if not is_training:
            # update dropout probability in inference mode
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0 
    
        # create embedding layer
        self.embedding = tf.keras.layers.Embedding(config.vocab_size, config.hidden_size, mask_zero=True, trainable=True)
        # create token type embeddings
        self.token_type_encoding = TokenTypeEncoding(config=config)
        # create positional embeddings
        self.pos_encoding = PositionalEncoding(config, self.seq_length, self.width)
        # add normalization layer
        self.norm_layer = tf.keras.layers.LayerNormalization(axis=-1)
        # create encoder
        self.enc_layers = EncoderLayer(config=config)
        # The "pooler" converts the encoded sequence tensor of shape
        # [batch_size, seq_length, hidden_size] to a tensor of shape
        # [batch_size, hidden_size]. This is necessary for segment-level
        # (or segment-pair-level) classification tasks where we need a fixed
        # dimensional representation of the segment.
        self.pooled_output = tf.keras.layers.Dense(config.hidden_size,activation=tf.tanh,
                            kernel_initializer=create_initializer(config.initializer_range))

        self.last_dense = tf.keras.layers.Dense(2, activation=tf.nn.log_softmax,
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))

        self.last_dense_prob = tf.keras.layers.Dense(2, activation=tf.nn.softmax,
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
      

    def __call__(self, input_ids, input_mask, token_type_ids):
        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, self.seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, self.seq_length], dtype=tf.int32)
        
        x = self.embedding(input_ids)

        token_type_embeddings = self.token_type_encoding(token_type_ids)
        token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, self.seq_length, self.width])
        x = x + token_type_embeddings
        x = x + self.pos_encoding
        x = self.norm_layer(x)
        x = dropout(x, self.dropout_prob)
        # x = x + self.norm_layer(x)  # maybe x = self.norm_layer(x)
        # x = x + dropout(x, self.dropout_prob)  # and x = dropout(x, self.dropout_prob)
        
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = create_attention_mask_from_input_mask(input_ids, input_mask)
        
        encoder_output = self.enc_layers(x, attention_mask, True, True)
        x = encoder_output[-1] # `sequence_output` shape = [batch_size, seq_length, hidden_size]
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(x[:, 0:1, :], axis=1)

        #Last layer hidden-state of the first token of the sequence (classification token) 
        #further processed by a Linear layer and a Tanh activation function.
        x = self.pooled_output(first_token_tensor) # [batch_size, hidden_size]


        # output_layer = model(input_ids, input_mask, token_type_ids)

        # hidden_size = output_layer.shape[-1]

        # weights_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)

        # output_weights = tf.Variable(initial_value=weights_initializer(shape=[2, self.width]), trainable=True,
        #     name="output_weights")

        # bias_initializer = tf.zeros_initializer()

        # output_bias = tf.Variable(initial_value=bias_initializer(shape=[2]), trainable=True,
        #     name="output_bias")

        # output_layer = tf.nn.dropout(x, rate=1-0.9)
        x = tf.nn.dropout(x, rate=1-0.9)  # [batch_size, hidden_size]
        # logits_2_1 = tf.linalg.matmul(logits_1, output_weights, transpose_b=True) # [batch_size, num_labels]
        # logits_2 = tf.nn.bias_add(logits_2_1, output_bias) # [batch_size, num_labels]
        # probabilities = tf.nn.softmax(logits_2, axis=-1)
        # log_probs_1 = tf.nn.log_softmax(logits_1, axis=-1) # [batch_size, hidden_size]
        # log_probs_2 = tf.nn.log_softmax(logits_2, axis=-1) # [batch_size, num_labels]
        log_x = self.last_dense(x) # [batch_size, num_labels]
        prob_x = self.last_dense_prob(x)
        # return x, logits_1, logits_2_1, logits_2, log_probs_1, log_probs_2, probabilities
        return log_x, prob_x


# def load_dataset(tfrecords, global_batch_size):
#     # Get list of TFRecord files
#     tfrecords = glob.glob(os.path.join(tfrecords, "*.tfrec"))
#     print(tfrecords)
#     dataset = tf.data.TFRecordDataset(tfrecords)
#     dataset = dataset.map(map_func=decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     dataset = dataset.cache()
#     dataset = dataset.shuffle(1000)
#     dataset = dataset.batch(global_batch_size)
#     dataset = dataset.prefetch(1)
#     return dataset


# def decode_fn(proto_example):
#     features = {
#         'input_ids': tf.io.FixedLenFeature([253], dtype=tf.int64, default_value=[0]*253),
#         'input_mask': tf.io.FixedLenFeature([253], dtype=tf.int64, default_value=[0]*253),
#         'segment_ids': tf.io.FixedLenFeature([253], dtype=tf.int64, default_value=[0]*253),
#         'label_ids': tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
#         'is_real_example': tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
#     }

#     # load one example
#     parsed_example = tf.io.parse_single_example(serialized=proto_example, features=features)
    
#     input_ids = parsed_example["input_ids"]
#     input_mask = parsed_example["input_mask"]
#     segment_ids = parsed_example["segment_ids"]
#     label_ids = parsed_example['label_ids']
#     is_real_example = parsed_example['is_real_example']
    
#     return input_ids, input_mask, segment_ids, label_ids


# define the DALI pipeline
@pipeline_def
def get_dali_pipeline(tfrec_filenames, tfrec_idx_filenames, initial_fill, training=True):
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
    # is_real_example = inputs['is_real_example'].gpu()

    return (input_ids, input_mask, segment_ids, label_ids)


# class DALIPreprocessor(object):
#     def __init__(self, filenames, idx_filenames, batch_size, vector_size, initial_fill,
#                deterministic=False, training=False):

#         device_id = hvd.local_rank()
#         shard_id = hvd.rank()
#         num_gpus = hvd.size()
#         self.pipe = get_dali_pipeline(tfrec_filenames=filenames, tfrec_idx_filenames=idx_filenames, batch_size=batch_size,
#                                       device_id=device_id, shard_id=shard_id, initial_fill=initial_fill, num_gpus=num_gpus,
#                                       training=training, seed=7 * (1 + hvd.rank()) if deterministic else None)

#         # self.daliop = dali_tf.DALIIterator()

#         self.batch_size = batch_size
#         self.device_id = device_id

#         self.dalidataset = dali_tf.DALIDataset(fail_on_device_mismatch=False, pipeline=self.pipe,
#             output_shapes=((batch_size, vector_size), (batch_size)),
#             batch_size=batch_size, output_dtypes=(tf.int64, tf.int64), device_id=device_id)

#     def get_device_dataset(self):
#         return self.dalidataset



# @tf.function
def training_step(data, num_labels, train_accuracy, loss, opt, model, first_batch):
    with tf.GradientTape() as tape:
        input_ids, input_mask, token_type_ids, labels = data

        # hidden_size = output_layer.shape[-1]

        # weights_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)

        # output_weights = tf.Variable(initial_value=weights_initializer(shape=[num_labels, hidden_size]), trainable=True,
        #     name="output_weights")

        # bias_initializer = tf.zeros_initializer()

        # output_bias = tf.Variable(initial_value=bias_initializer(shape=[num_labels]), trainable=True,
        #     name="output_bias")

        # output_layer = tf.nn.dropout(output_layer, rate=1-0.9)

        # logits_1 = tf.linalg.matmul(output_layer, output_weights, transpose_b=True)
        # logits_2 = tf.nn.bias_add(logits_1, output_bias)
        # probabilities = tf.nn.softmax(logits_2, axis=-1)
        # log_probs = tf.nn.log_softmax(logits_2, axis=-1)
        log_probs, probabilities = model(input_ids, input_mask, token_type_ids)
        # x, logits_1, logits_2_1, logits_2, log_probs_1, log_probs_2, probabilities = model(input_ids, input_mask, token_type_ids)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        product = one_hot_labels * log_probs
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss_value = tf.reduce_mean(per_example_loss)

    grads = tape.gradient(loss_value, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    #update training accuracy
    train_accuracy.update_state(labels, probabilities)
    return log_probs, probabilities, one_hot_labels, loss_value, product, per_example_loss
    # return log_probs, grads, loss_value 
    # return loss_value, probabilities
    # return loss_value, probabilities, logits_1, logits_2, log_probs, one_hot_labels, per_example_loss, per_example_loss_1


def main():
  global_batch_size = 256
  output_dir = os.getcwd()
  rnd = 1
  vector_size = 253
  initial_fill = 1000
  epochs = 50
  num_train_examples = int(sys.argv[1]) # 632118
  nstep_per_epoch = num_train_examples // global_batch_size
  num_train_steps = int(nstep_per_epoch * epochs)  # total number of training steps/batches
  print(f'num_train_examples: {num_train_examples}\nnstep_per_epoch : {nstep_per_epoch }\nnum_train_steps: {num_train_steps}')
  tfrecords = str(sys.argv[2])
  # tfrecords = "/nese/zhanglab/ccres/archive/cecile_cres_uri_edu-dl-toda/129-data/bert/train-tfrecords/tfrecords-bert-finetuning"
  # dataset = load_dataset(tfrecords, global_batch_size)

  train_files = sorted(glob.glob(os.path.join(tfrecords, 'train*.tfrec')))
  train_idx_files = sorted(glob.glob(os.path.join(os.path.join(tfrecords, 'idx_files'), 'train*.idx')))


  dataset = dali_tf.DALIDataset(pipeline=get_dali_pipeline(tfrec_filenames=train_files, tfrec_idx_filenames=train_idx_files, 
                                    initial_fill=initial_fill, batch_size=global_batch_size, training=True), output_shapes=((global_batch_size, vector_size), (global_batch_size, vector_size), (global_batch_size, vector_size), (global_batch_size)),
                                output_dtypes=(tf.int64, tf.int64, tf.int64, tf.int64), batch_size=global_batch_size, num_threads=4, device_id=0)
                                

  bert_config_file = '/nese/zhanglab/ccres/archive/cecile_cres_uri_edu-dl-toda/bert_tf2/bert_config.json'
  bert_config = BertConfig.from_json_file(bert_config_file)
  print(bert_config)

  num_labels = 2

  is_training = True
  model = BertModel(
        config=bert_config,
        is_training=is_training)

  for var in model.variables:
    print(var, "\n")

  print(f'# variables: {len(model.variables)}')



  

  # define metrics
  loss = tf.losses.SparseCategoricalCrossentropy()
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

  init_lr = 5e-5

  # define learning rate polynomial decay
  # linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
  #   initial_learning_rate=init_lr,
  #   end_learning_rate=0,
  #   decay_steps=num_train_steps)

  # define linear warmup schedule
  # warmup_proportion = 0.1  # Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training
  # warmup_steps = int(warmup_proportion * num_train_steps)
  # warmup_schedule = tfm.optimization.lr_schedule.LinearWarmup(
  #   warmup_learning_rate = 0,
  #   after_warmup_lr_sched = linear_decay,
  #   warmup_steps = warmup_steps)

  # define optimizer
  # opt = tf.keras.optimizers.Adam(learning_rate=warmup_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-6, weight_decay=0.01)
  opt = tf.keras.optimizers.Adam(learning_rate=init_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-6, weight_decay=0.01)
  # exclude variables from weight decay
  opt.exclude_from_weight_decay(var_names=["LayerNorm", "layer_norm", "bias"])

  # if hvd.rank() == 0:
  # create output directory
  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

  # create directory for storing checkpoints
  ckpt_dir = os.path.join(output_dir, f'ckpts-rnd-{rnd}')
  if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)

  # create checkpoint object to save model
  checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)

  # create directory for storing logs
  tensorboard_dir = os.path.join(output_dir, f'logs-rnd-{rnd}')
  if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)

  writer = tf.summary.create_file_writer(tensorboard_dir)
  td_writer = open(os.path.join(output_dir, f'logs-rnd-{rnd}', f'training_data_rnd_{rnd}.tsv'), 'w')
  vd_writer = open(os.path.join(output_dir, f'logs-rnd-{rnd}', f'validation_data_rnd_{rnd}.tsv'), 'w')

  start = datetime.datetime.now()

  epoch = 0

  for batch, data in enumerate(dataset.take(nstep_per_epoch*epochs), 1):
    input_ids, input_mask, token_type_ids, labels = data

    # print(input_ids, input_mask, token_type_ids, labels)
    # output_layer = training_step(data, num_labels, train_accuracy, loss, opt, model, batch == 1)
    # print(output_layer)
    log_probs, probabilities, one_hot_labels, loss_value, product, per_example_loss = training_step(data, num_labels, train_accuracy, loss, opt, model, batch == 1)
    print(log_probs, probabilities, one_hot_labels, loss_value, product, per_example_loss)
    break
    # log_probs, grads, loss_value = training_step(data, num_labels, train_accuracy, loss, opt, model, batch == 1)
    # loss_value, probs, logits_1, logits_2, log_probs, one_hot_labels, per_example_loss, per_example_loss_1  = training_step(data, num_labels, train_accuracy, loss, opt, model, batch == 1)
    # break
    
    # print(f'logits 1: {logits_1}')
    # print(f'logits 2: {logits_2}')
    # print(f'probabilities: {probs}')
    # print(f'log_probs: {log_probs}')
    # print(f'one_hot_labels: {one_hot_labels}')
    # print(f'per_example_loss: {per_example_loss}')
    # print(f'per_example_loss_1: {per_example_loss_1}')
    # print(f'loss_value: {loss_value}')

    # if batch % 100 == 0 and hvd.rank() == 0:
    if batch == 1:
      # with open(os.path.join(output_dir, f'model-bert.txt'), 'w+') as f:
      #   model.summary(print_fn=lambda x: f.write(x + '\n')) 
      print(f'# trainable variables: {len(model.trainable_variables)}')
    if batch % 10 == 0 :
      # print(f'grads: {grads}')
      # print(input_ids, input_mask, token_type_ids, labels)
      print(f'Epoch: {epoch} - Step: {batch} - learning rate: {opt.learning_rate.numpy()} - Training loss: {loss_value} - Training accuracy: {train_accuracy.result().numpy()*100}')
      # write metrics
      with writer.as_default():
          tf.summary.scalar("learning_rate", opt.learning_rate, step=batch)
          tf.summary.scalar("train_loss", loss_value, step=batch)
          tf.summary.scalar("train_accuracy", train_accuracy.result().numpy(), step=batch)
          writer.flush()
      td_writer.write(f'{epoch}\t{batch}\t{opt.learning_rate.numpy()}\t{loss_value}\t{train_accuracy.result().numpy()}\n')
    if batch % 50 == 0 :
      break

    if batch % nstep_per_epoch == 0:
      epoch += 1

  

if __name__ == "__main__":
  main()
