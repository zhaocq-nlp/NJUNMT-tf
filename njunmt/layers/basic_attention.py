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

from njunmt.utils.configurable import Configurable

FLOAT_MIN = -1.e9


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

        Raises:
            NotImplementedError: if `query_is_projected` and `key_is_projected`
              are both False, or `memory_bias` is provided, or `memory_length`
              is not provided.
        """
        if not query_is_projected or not key_is_projected:
            raise NotImplementedError("query and key should be pre-projected before build fn is called")
        if memory_length is None or memory_bias is not None:
            raise NotImplementedError("in BaseAttention, we only use values_length to mask the attention values")

        # expanded query: [batch_size, 1, channels_query]
        # energies: [batch_size, num_of_keys]
        energies = self.att_fn(query, keys)

        # Replace all scores for padded inputs with tf.float32.min
        num_scores = tf.shape(energies)[1]  # number of keys
        scores_mask = tf.sequence_mask(
            lengths=tf.to_int32(memory_length),
            maxlen=tf.to_int32(num_scores),
            dtype=tf.float32)

        energies = energies * scores_mask + ((1.0 - scores_mask) * FLOAT_MIN)

        # Stabilize energies first and then exp
        energies = energies - tf.reduce_max(energies, axis=1, keep_dims=True)
        unnormalized_scores = tf.exp(energies) * scores_mask

        normalization = tf.reduce_sum(unnormalized_scores, axis=1, keep_dims=True)

        # Normalize the scores
        # [batch_size, num_of_values]
        scores_normalized = unnormalized_scores / normalization

        # Calculate the weighted average of the attention inputs
        # according to the scores
        #   [batch_size, num_of_values, 1] * [batch_size, num_of_values, channels_value]
        context = tf.expand_dims(scores_normalized, 2) * memory
        #   [batch_size, channels_value]
        context = tf.reduce_sum(context, 1, name="context")
        context.set_shape([None, memory.get_shape().as_list()[-1]])

        return scores_normalized, context


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
        return {"num_units": 512}

    def att_fn(self, query, keys, bias=None):
        """ Computes attention energies using a feed forward layer,
        which will be passed into a softmax function.

        Args:
            query: Attention query tensor with shape
              [batch_size, channels_query]
            keys: Attention keys tensor with shape
              [batch_size, num_of_keys, channels_key]
            bias: The bias tensor for attention keys

        Returns: A Tensor, [batch_size, num_of_keys]
        """
        with tf.variable_scope(self.name):
            v_att = tf.get_variable("v_att", shape=[self.params["num_units"]], dtype=tf.float32)
        return tf.reduce_sum(v_att * tf.tanh(keys + tf.expand_dims(query, 1)), [2])
