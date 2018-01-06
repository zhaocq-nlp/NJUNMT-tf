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
""" Define base class for attention and BahdanauAttention class. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod, abstractproperty
import tensorflow as tf

from njunmt.layers.common_layers import fflayer
from njunmt.layers.common_layers import dropout_wrapper
from njunmt.utils.configurable import Configurable
from njunmt.utils.algebra_ops import advanced_softmax

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


def attention_bias_ignore_padding(memory_padding):
    """ Create a bias tensor to be added to attention logits

    Args:
        memory_padding: A float Tensor with shape [batch_size, memory_length],
          where 1.0 for padding and 0.0 for non-padding.

    Returns: A float Tensor with shape [batch_size, memory_length],
      where -1e9 for padding and 0 for non-padding.
    """
    return FLOAT_MIN * memory_padding


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
                    memory_bias = attention_bias_ignore_padding(memory_padding)

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
