# Copyright 2017 Natural Language Processing Group, Nanjing University, zhaocq.nlp@gmail.com.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Define base class for attention, BahdanauAttention class
and MultiHeadAttention class and functions. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod, abstractproperty
import tensorflow as tf

from njunmt.layers.common_layers import fflayer
from njunmt.layers.common_layers import conv1d
from njunmt.layers.common_layers import dropout_wrapper
from njunmt.utils.configurable import Configurable
from njunmt.utils.algebra_ops import advanced_softmax
from njunmt.utils.algebra_ops import split_last_dimension
from njunmt.utils.algebra_ops import combine_last_two_dimensions

FLOAT_MIN = -1.e9


def embedding_to_padding(emb, sequence_length):
    """ Calculates the padding mask based on `sequence_length`.

    Args:
        emb: An input embedding `Tensor` with shape
          [batch_size, maximum_sequence_length, dmodel]
        sequence_length: Length of each sequence in `emb`,
           a Tensor with shape [batch_size, ]

    Returns: A float Tensor with shape [batch_size, maximum_sequence_length],
      where 1.0 for padding and 0.0 for non-padding.
    """
    if emb is None:
        seq_mask = 1. - tf.sequence_mask(
            lengths=tf.to_int32(sequence_length),
            maxlen=tf.reduce_max(sequence_length),
            dtype=tf.float32)  # 1.0 for padding
    else:
        seq_mask = 1. - tf.sequence_mask(
            lengths=tf.to_int32(sequence_length),
            maxlen=tf.shape(emb)[1],
            dtype=tf.float32)  # 1.0 for padding
    return seq_mask


class BaseAttention(Configurable):
    """ Define base attention class. """

    def __init__(self, params, mode, name=None):
        """ Initializes the parameters of the attention.

        Args:
            params: A dictionary of parameters to construct the
              decoder architecture.
            mode: A mode.
            name: The name of this attention.
        """
        super(BaseAttention, self).__init__(
            params=params, mode=mode, verbose=False,
            name=name or self.__class__.__name__)

    @abstractproperty
    def attention_units(self):
        """ Returns the number of units of this attention mechanism. """
        raise NotImplementedError

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this attention. """
        raise NotImplementedError

    @abstractmethod
    def att_fn(self, query, keys, bias):
        """ Computes attention energies, which will be passed into a
        softmax function.

        Args:
            query: Attention query tensor with shape
              [batch_size, channels_query]
            keys: Attention keys tensor with shape
              [batch_size, num_of_keys, channels_key]
            bias: The bias tensor for attention keys

        Returns: A Tensor, [batch_size, num_of_keys]
        """
        raise NotImplementedError

    def build(self,
              query,
              keys,
              memory,
              memory_length=None,
              memory_bias=None,
              query_is_projected=True,
              key_is_projected=True):
        """ Builds attention context via a simple process.

        Args:
            query: Attention query tensor with shape
              [batch_size, channels_query].
            keys: Attention keys tensor with shape
              [batch_size, num_of_keys, channels_key].
            memory: Attention values tensor with shape
              [batch_size, num_of_values, channels_value].
            memory_length: The number of attention values, a
              Tensor with shape [batch_size,].
            memory_bias: The bias tensor for attention values,
              not used here.
            query_is_projected: Whether the `query` is already projected.
            key_is_projected: Whether the `keys` is already projected.

        Returns: A tuple `(attention_scores, attention_context)`. The
          `attention_scores` has shape [batch_size, num_of_values].
          The `attention_context` has shape [batch_size, channels_value].
        """
        with tf.variable_scope(self.name):
            if not query_is_projected:
                query = fflayer(query, output_size=self.attention_units,
                                activation=None, name="ff_att_query")
            if not key_is_projected:
                keys = fflayer(keys, output_size=self.attention_units,
                               activation=None, name="ff_att_keys")

            if memory_bias is None:
                if memory_length is not None:
                    memory_padding = embedding_to_padding(memory, memory_length)
                    memory_bias = memory_padding * FLOAT_MIN

            # attention weights: [batch_size, num_of_values]
            attention_weight = self.att_fn(query, keys, memory_bias)

            # Calculate the weighted average of the attention inputs
            # according to the scores
            #   [batch_size, num_of_values, 1] * [batch_size, num_of_values, channels_value]
            context = tf.expand_dims(attention_weight, 2) * memory
            #   [batch_size, channels_value]
            context = tf.reduce_sum(context, 1, name="context")
            context.set_shape([None, memory.get_shape().as_list()[-1]])

            return attention_weight, context


class BahdanauAttention(BaseAttention):
    """ Attention mechanism described in https://arxiv.org/abs/1409.0473. """

    def __init__(self, params, mode, name=None):
        """ Initializes the parameters of the attention.

        Args:
            params: A dictionary of parameters to construct the
              decoder architecture.
            mode: A mode.
            name: The name of this attention.
        """
        super(BahdanauAttention, self).__init__(params, mode, name)

    @property
    def attention_units(self):
        """ Returns the number of units of this attention mechanism. """
        return self.params["num_units"]

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this attention. """
        return {"num_units": 512,
                "dropout_attention_keep_prob": 1.0}

    def att_fn(self, query, keys, bias=None):
        """ Computes attention scores.

        Args:
            query: Attention query tensor with shape
              [batch_size, channels_query]
            keys: Attention keys tensor with shape
              [batch_size, num_of_keys, channels_key]
            bias: The bias tensor for attention keys

        Returns: A Tensor, [batch_size, num_of_keys]
        """
        v_att = tf.get_variable("v_att", shape=[self.params["num_units"]], dtype=tf.float32)
        logits = tf.reduce_sum(v_att * tf.tanh(keys + tf.expand_dims(query, 1)), [2])
        if bias is not None:
            logits += bias
        attention_scores = advanced_softmax(logits)
        attention_scores = dropout_wrapper(attention_scores, self.params["dropout_attention_keep_prob"])
        return attention_scores


def attention_bias_ignore_padding(memory_padding):
    """ Create a bias tensor to be added to attention logits

    Args:
        memory_padding: A float Tensor with shape [batch_size, memory_length],
          where 1.0 for padding and 0.0 for non-padding.

    Returns: A float Tensor with shape [batch_size, 1, 1, memory_length],
      where -1e9 for padding and 0 for non-padding.
    """
    ret = FLOAT_MIN * memory_padding
    return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)


def attention_bias_to_padding(attention_bias):
    """ Inverse of attention_bias_ignore_padding()

    Args:
        attention_bias: A float Tensor with shape [batch_size, 1, 1, memory_length],
          as returned by attention_bias_ignore_padding().

    Returns: A Tensor with shape [batch, memory_length] with 1.0 in padding positions
      and 0.0 in non-padding positions.
    """
    return tf.squeeze(tf.to_float(tf.less(attention_bias, -1)),
                      axis=[1, 2])


def attention_bias_lower_triangle(length):
    """ Create a bias tensor to be added to attention logits.

      Allows a query to attend to all positions up to and including its own.
    Args:
        length: A scalar.

    Returns: A float Tensor of shape [1, 1, length, length], with -1e9 in
      padding positions and 0 in non-padding positions.

    """
    lower_triangle = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
    ret = FLOAT_MIN * (1. - lower_triangle)
    return tf.reshape(ret, [1, 1, length, length])


def dot_product_attention(q, k, bias=None, dropout_keep_prob=1.0):
    """ Computes attention weight according to query and key.

    Args:
        q: A query Tensor with shape [..., length_q, depth].
        k: A keys Tensor with shape [..., length_k, depth].
        bias: A bias Tensor with shape [..., 1, depth].
        dropout_keep_prob: A float scalar.

    Returns: The attention scores Tensor with shape
      [..., length_q, length_k].
    """
    with tf.variable_scope("dot_product_attention", values=[q, k]):
        logits = tf.matmul(q, k, transpose_b=True)
        if bias is not None:
            logits += bias
        weights = advanced_softmax(logits)
        # dropout the attention links for each of the heads
        weights = dropout_wrapper(weights, keep_prob=dropout_keep_prob)
        return weights


def multihead_attention_layer(params,
                              mode,
                              query_antecedent,
                              memory_antecedent,
                              memory_bias,
                              name=None):
    """ Multi-head scaled-dot-product attention with input/output
      transformations.

    Args:
        params: A dictionary of parameters to construct the
          attention architecture.
        mode: A mode.
        query_antecedent: A Tensor with shape [batch_size, length_q, channels_query],
          if not provided (None), means self attention and `query_antecedent` will be
          the same as `memory_antecedent`.
        memory_antecedent: A Tensor with shape [batch_size, length_m, channels_memory]
        memory_bias: A bias Tensor for `memory_antecedent`.
        name: The name of this attention.

    Returns: The result of the attention transformation. The output shape is
      [batch_size, length_q, hidden_dim].
    """
    cls = MultiHeadAttention(params, mode, name)
    return cls.build(
        query=query_antecedent,
        keys=None,
        memory=memory_antecedent,
        memory_bias=memory_bias,
        key_is_projected=False,
        query_is_projected=False)


class MultiHeadAttention(BaseAttention):
    """ Class of multi-head scaled-dot-product attention with input/output
      transformations.
    """

    def __init__(self, params, mode, name=None):
        """ Initializes the parameters of the attention.

        Args:
            params: A dictionary of parameters to construct the
              decoder architecture.
            mode: A mode.
            name: The name of this attention.
        """
        super(MultiHeadAttention, self).__init__(params, mode, name)
        self._num_heads = self.params["num_heads"]
        self._num_units = self.params["num_units"]
        self._attention_key_depth = self.params["attention_key_depth"] or self._num_units
        self._attention_value_depth = self.params["attention_value_depth"] or self._num_units
        self._output_depth = self.params["output_depth"] or self._num_units
        self._dropout_attention_keep_prob = self.params["dropout_attention_keep_prob"]
        self._attention_type = self.params["attention_type"]
        if self._attention_key_depth % self._num_heads != 0:
            raise ValueError("query depth ({}) must be divisible by the number of "
                             "attention heads ({}).".format(self._attention_key_depth, self._num_heads))
        if self._attention_value_depth % self._num_heads != 0:
            raise ValueError("value depth ({}) must be divisible by the number of "
                             "attention heads ({}).".format(self._attention_value_depth, self._num_heads))

    def _check_parameters(self):
        assert self.params["attention_type"] in ["dot_product"], (
            "only attention_type=\"dot_product\" is available in MultiHeadAttention")

    @property
    def attention_units(self):
        """ Returns the number of units of this attention mechanism. """
        return self.params["num_units"]

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this attention. """
        return {
            "num_heads": 8,
            "num_units": 512,
            "attention_key_depth": None,
            "attention_value_depth": None,
            "output_depth": None,
            "dropout_attention_keep_prob": 0.9,
            "attention_type": "dot_product"
        }

    def build(self,
              query,
              keys,
              memory,
              memory_length=None,
              memory_bias=None,
              query_is_projected=False,
              key_is_projected=False):
        """ Builds attention context.

        Args:
            query: Attention query tensor with shape [batch_size, length_q, channels_query].
              If None, it indicates self-attention.
            keys: Attention keys tensor with shape [batch_size, length_k, channels_key].
            memory: Attention values tensor with shape [batch_size, length_m, channels_value].
            memory_length: The number of attention values, [batch_size,].
            memory_bias: The bias tensor for attention values with shape [batch_size, 1, 1, timesteps].
            query_is_projected: Whether the `query` is already projected.
            key_is_projected: Whether the `keys` is already projected.

        Returns: The result of the attention transformation. A tuple
        `(attention_scores, attention_context)`. The `attention_scores`
        has shape [batch_size, num_heads, length_q, length_k]. The
        `attention_context` has shape [batch_size, length_q, output_depth].
        """
        with tf.variable_scope(self.name, values=[query, memory]):
            query_is_2d = False
            if query is not None and query.get_shape().ndims == 2:
                # for using MultiHeadAttention in RNN-based decoders
                query_is_2d = True
                query = tf.expand_dims(query, axis=1)
            # compute q, k, v
            q, k, v = self._compute_qkv(query, keys, memory,
                                        query_is_projected=query_is_projected,
                                        key_is_projected=key_is_projected)

            # after split_last_dimension: [batch_size, length, depth]
            #           ==> [batch_size, length, num_heads, depth/num_heads]
            # after split_head: ==> [batch_size, num_heads, length, depth/num_heads]
            split_head = lambda _x, _nh: tf.transpose(split_last_dimension(_x, _nh),
                                                      [0, 2, 1, 3])
            # [batch_size, num_heads, length, depth/num_heads] ==> [batch_size, length, depth]
            combine_head = lambda _x: combine_last_two_dimensions(
                tf.transpose(_x, [0, 2, 1, 3]))

            # [batch_size, num_heads, length_q/k/v, depth/num_heads]
            q = split_head(q, self._num_heads)
            k = split_head(k, self._num_heads)
            v = split_head(v, self._num_heads)
            # last dim of q, k, v after split_head
            key_depth_per_head = self._attention_key_depth // self._num_heads
            q *= key_depth_per_head ** (-0.5)  # scale the query

            if memory_bias is None:
                if memory_length is not None:
                    assert query is not None, "Unseen error may occur. Please CHECK."
                    input_padding = embedding_to_padding(memory, memory_length)
                    # [batch_size, 1, 1, timesteps], FLOAT_MIN for padding, 0.0 for non-padding
                    memory_bias = attention_bias_ignore_padding(input_padding)

            # compute attention weight, [batch_size, num_heads, length_q, length_k]
            attention_weight = self.att_fn(q, k, memory_bias)
            # sum over attention values, [batch_size, num_heads, length_q, depth/num_heads]
            attention_context = tf.matmul(attention_weight, v)

            # combined: [batch_size, length_q, depth_value]
            attention_context = combine_head(attention_context)
            # linear transform
            attention_context = conv1d(attention_context, self._output_depth, kernel_size=1,
                                       name="output_transform")
            # TODO here the dimension of attention_weight is not checked for output_attention
            if query_is_2d:
                attention_context = tf.squeeze(attention_context, axis=1)
            return attention_weight, attention_context

    def _compute_qkv(self, query, keys, memory,
                     query_is_projected=False, key_is_projected=False):
        """ Computes linear transformations of query, key
         and value.

        Args:
            query: Attention query tensor with shape [batch_size, length_q, channels_query].
              If None, it indicates self-attention.
            keys: Attention keys tensor with shape [batch_size, length_k, channels_key].
            memory: Attention values tensor with shape
              [batch_size, length_m, channels_value]
            query_is_projected: Whether the `query` is already projected.
            key_is_projected: Whether the `keys` is already projected.
        Returns: A tuple `(query_transformed, key_transformed,
          memory_transformed)`.
        """
        if query is None:
            # indicating self-attention, query and key are both the same as memory
            _ = keys
            combined = conv1d(
                memory,
                self._attention_key_depth * 2 + self._attention_value_depth,
                kernel_size=1, name="qkv_transform")
            q, k, v = tf.split(
                combined,
                [self._attention_key_depth, self._attention_key_depth,
                 self._attention_value_depth],
                axis=2)
            return q, k, v
        else:
            # encoder-decoder attention
            if query_is_projected:
                q = query
            else:
                q = conv1d(
                    query,
                    self._attention_key_depth,
                    kernel_size=1,
                    name="q_transform",
                    padding="VALID")
            if key_is_projected:
                k = keys
                v = conv1d(memory,
                           self._attention_value_depth,
                           kernel_size=1,
                           name="v_transform",
                           padding="VALID")
            else:
                kv_combined = conv1d(
                    memory,
                    self._attention_key_depth + self._attention_value_depth,
                    kernel_size=1,
                    name="kv_transform",
                    padding="VALID")
                k, v = tf.split(
                    kv_combined,
                    [self._attention_key_depth, self._attention_value_depth],
                    axis=2)
            return q, k, v

    def att_fn(self, q, k, bias):
        """ Computes attention scores according to attention_type.

        Args:
            q: Attention query tensor with shape
              [batch_size, num_heads, length_q, depth / num_heads]
            k: Attention keys tensor with shape
              [batch_size, num_heads, length_k, depth / num_heads]
            bias: The bias tensor for attention keys with shape
              [batch_size, 1, 1, length_k]

        Returns: A Tensor, [batch_size, num_heads, length_q, length_k]
        """
        if self._attention_type == "dot_product":
            x = dot_product_attention(
                q, k, bias=bias, dropout_keep_prob=self._dropout_attention_keep_prob)
        else:
            raise NotImplementedError("att_fn for \"{}\" not implemented.".format(self._attention_type))

        return x
