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
""" Define RNN-based decoders. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys
from collections import namedtuple

import tensorflow as tf

from njunmt.decoders.decoder import Decoder
from njunmt.decoders.decoder import initialize_cache
from njunmt.layers import common_attention
from njunmt.layers.common_layers import fflayer
from njunmt.utils.rnn_cell_utils import get_condr_rnn_cell
from njunmt.utils.rnn_cell_utils import get_multilayer_rnn_cells

# import all attention
ATTENTION_CLS = [
    x for x in common_attention.__dict__.values()
    if inspect.isclass(x) and issubclass(x, common_attention.BaseAttention)
    ]
for att in ATTENTION_CLS:
    setattr(sys.modules[__name__], att.__name__, att)


class CondAttentionDecoder(Decoder):
    """ Define an RNN Decoder that uses attention over an input sequence.
    It use conditional LSTM/GRU as a base layer. The following layers are
    Stacked rGRU/rLSTM described in https://arxiv.org/abs/1707.07631.
    """

    def __init__(self,
                 params,
                 mode,
                 name=None,
                 verbose=True):
        """ Initializes decoder parameters.

        Args:
            params: A dictionary of parameters to construct the
              decoder architecture.
            mode: A mode.
            name: The name of this decoder.
            verbose: Print decoder parameters if set True.
        """
        super(CondAttentionDecoder, self).__init__(params, mode, name, verbose)

        # recurrent cells
        self._cond_rnn_cell, self._r_rnn_cells = get_condr_rnn_cell(**self.params['rnn_cell'])
        # for attention
        attention_cls = self.params["attention.class"]
        self._attention = eval(attention_cls)(
            self.params["attention.params"], mode=self.mode)

        self._DecoderOutputSpec = namedtuple(
            "DecoderOutput",
            "cur_decoder_hidden prev_input attention_context attention_scores")

    @property
    def output_ignore_fields(self):
        """ Returns a list of strings. The loop in `dynamic_decode`
        function will not save these fields in `output_dtype` during
        inference, for the sake of reducing device memory.
        """
        return ["cur_decoder_hidden", "prev_input", "attention_context"]

    @property
    def output_dtype(self):
        """ Returns a `collections.namedtuple`,
        the definition of decoder output types. """
        return self._DecoderOutputSpec(
            cur_decoder_hidden=tf.float32,
            prev_input=tf.float32,
            attention_context=tf.float32,
            attention_scores=tf.float32)

    def _check_parameters(self):
        assert self.params["attention.class"] in \
               ["BahdanauAttention", "MultiHeadAttention"], (
            "attention.class should be BahdanauAttention or MultiHeadAttention")

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this decoder. """
        return {
            "attention.class": "BahdanauAttention",
            "attention.params": {},  # Arbitrary attention layer parameters
            "rnn_cell": {
                "cell_class": "LSTMCell",
                "cell_params": {
                    "num_units": 1024
                },
                "dropout_input_keep_prob": 1.0,
                "dropout_state_keep_prob": 1.0,
                "num_layers": 1,
            },
            "dropout_context_keep_prob": 1.0,
            "dropout_hidden_keep_prob": 1.0,
            "dropout_embedding_keep_prob": 1.0,
            "logits_dimension": 512
        }

    def prepare(self, encoder_output, bridge, helper):
        """ Prepares for `step()` function.
        Do:
            1. initialize decoder RNN states using `bridge`;
            2. acquire attention information from `encoder_output`;
            3. pre-project the attention keys

        Args:
            encoder_output: An instance of `collections.namedtuple`
              from `Encoder.encode()`.
            bridge: An instance of `Bridge` that initializes the
              decoder states.
            helper: An instance of `Feedback` that samples next
              symbols from logits.
        Returns: A dict containing decoder RNN states, pre-projected attention
          keys, attention values and attention length, and will be passed
          to `step()` function.
        """
        attention_values = encoder_output.attention_values
        if hasattr(encoder_output, "attention_bias"):
            attention_bias = encoder_output.attention_bias
        else:
            attention_length = encoder_output.attention_length
            attention_bias = getattr(eval(self.params["attention.class"]),
                                     "attention_length_to_bias")(None, attention_length)
        with tf.variable_scope(self._attention.name):
            projected_attention_keys = fflayer(
                inputs=attention_values, output_size=self._attention.attention_units,
                dropout_input_keep_prob=self.params["dropout_context_keep_prob"],
                activation=None, name="ff_att_keys")
        init_rnn_states = bridge(self._r_rnn_cells.state_size)
        init_cache = initialize_cache(
            decoding_states=init_rnn_states,
            attention_keys=projected_attention_keys,
            memory=attention_values,
            memory_bias=attention_bias)

        return init_cache

    def merge_top_features(self, decoder_output):
        """ Merges features of decoder top layers, as the input
        of softmax layer.
        Features to be merged are as follows:
            1. current decoder RNN state;
            2. current attention context;
            3. previous predicted word.

        Args:
            decoder_output: An instance of `collections.namedtuple`
              whose element types are defined by `output_dtype`
              property.

        Returns: A instance of `tf.Tensor`, as the input of
          softmax layer.
        """
        assert isinstance(decoder_output, self._DecoderOutputSpec)
        cur_decoder_hidden = decoder_output.cur_decoder_hidden
        prev_input = decoder_output.prev_input
        attention_context = decoder_output.attention_context

        logit_lstm = fflayer(cur_decoder_hidden, output_size=self.params["logits_dimension"],
                             dropout_input_keep_prob=self.params["dropout_hidden_keep_prob"],
                             activation=None, name="ff_logit_lstm")
        logit_prev = fflayer(prev_input, output_size=self.params["logits_dimension"],
                             dropout_input_keep_prob=self.params["dropout_embedding_keep_prob"],
                             activation=None, name="ff_logit_prev")
        logit_ctx = fflayer(attention_context, output_size=self.params["logits_dimension"],
                            dropout_input_keep_prob=self.params["dropout_hidden_keep_prob"],
                            activation=None, name="ff_logit_ctx")
        merged_output = tf.tanh(logit_lstm + logit_prev + logit_ctx)
        return merged_output

    def step(self, decoder_input, cache):
        """ Decodes one step.

        Args:
            decoder_input: The decoder input for this timestep, an
              instance of `tf.Tensor`, [batch_size, dim_word].
            cache: A dict containing decoder RNN states at previous
              timestep, pre-projected attention keys, attention values
              and attention length.

        Returns: A tuple `(cur_decoder_outputs, cur_cache)` at this timestep.
          The `cur_decoder_outputs` must be an instance of `collections.namedtuple`
          whose element types are defined by `output_dtype` property. The
          `cur_cache` must have the same structure with `cache`.
        """
        # layer0: get hidden1
        cell_output0, cell_state0 = self._cond_rnn_cell(decoder_input, cache["decoding_states"][0])

        # compute attention using hidden1
        # [batch_size, n_timesteps_src]
        attention_scores, attention_context = self._attention.build(
            query=cell_output0,
            memory=cache["memory"],
            memory_bias=cache["memory_bias"],
            cache=cache)
        # hidden1's state is the hidden2 's initial state
        following_decoder_state = tuple([cell_state0] + list(cache["decoding_states"][1:]))
        cell_output, cell_states = self._r_rnn_cells(
            attention_context, following_decoder_state)

        outputs = self._DecoderOutputSpec(
            cur_decoder_hidden=cell_output,
            prev_input=decoder_input,
            attention_context=attention_context,
            attention_scores=attention_scores)

        cache["decoding_states"] = cell_states
        return outputs, cache


class AttentionDecoder(Decoder):
    """ A simple RNN Decoder that uses attention over an input sequence.
    """

    def __init__(self,
                 params,
                 mode,
                 name=None,
                 verbose=True):
        """ Initializes decoder parameters.

        Args:
            params: A dictionary of parameters to construct the
              decoder architecture.
            mode: A mode.
            name: The name of this decoder.
            verbose: Print decoder parameters if set True.
        """
        super(AttentionDecoder, self).__init__(params, mode, name, verbose)

        # recurrent cells
        self._rnn_cells = get_multilayer_rnn_cells(**self.params['rnn_cell'])
        # for attention
        attention_cls = self.params["attention.class"]
        self._attention = eval(attention_cls)(
            self.params["attention.params"], mode=self.mode)

        self._DecoderOutputSpec = namedtuple(
            "DecoderOutput",
            "cur_decoder_hidden prev_input attention_context attention_scores")

    @property
    def output_ignore_fields(self):
        """ Returns a list of strings. The loop in `dynamic_decode`
        function will not save these fields in `output_dtype` during
        inference, for the sake of reducing device memory.
        """
        return ["cur_decoder_hidden", "prev_input", "attention_context"]

    @property
    def output_dtype(self):
        """ Returns a `collections.namedtuple`,
        the definition of decoder output types. """
        return self._DecoderOutputSpec(
            cur_decoder_hidden=tf.float32,
            prev_input=tf.float32,
            attention_context=tf.float32,
            attention_scores=tf.float32)

    def _check_parameters(self):
        assert self.params["attention.class"] in \
               ["BahdanauAttention", "MultiHeadAttention"], (
            "attention.class should be BahdanauAttention or MultiHeadAttention")

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this decoder. """
        return {
            "attention.class": "BahdanauAttention",
            "attention.params": {},  # Arbitrary attention layer parameters
            "rnn_cell": {
                "cell_class": "LSTMCell",
                "cell_params": {
                    "num_units": 1024
                },
                "dropout_input_keep_prob": 1.0,
                "dropout_state_keep_prob": 1.0,
                "num_layers": 1,
            },
            "dropout_context_keep_prob": 1.0,
            "dropout_hidden_keep_prob": 1.0,
            "dropout_embedding_keep_prob": 1.0,
            "logits_dimension": 512,
        }

    def prepare(self, encoder_output, bridge, helper):
        """ Prepares for `step()` function.
        Do,
            1. initialize decoder RNN states using `bridge`;
            2. acquire attention information from `encoder_output`;
            3. pre-project the attention keys

        Args:
            encoder_output: An instance of `collections.namedtuple`
              from `Encoder.encode()`.
            bridge: An instance of `Bridge` that initializes the
              decoder states.
            helper: An instance of `Feedback` that samples next
              symbols from logits.
        Returns: A dict containing decoder RNN states, pre-projected attention
          keys, attention values and attention length, and will be passed
          to `step()` function.
        """
        attention_values = encoder_output.attention_values  # [batch_size, timesteps, dim_context]
        if hasattr(encoder_output, "attention_bias"):
            attention_bias = encoder_output.attention_bias
        else:
            attention_length = encoder_output.attention_length
            attention_bias = getattr(eval(self.params["attention.class"]),
                                     "attention_length_to_bias")(None, attention_length)
        with tf.variable_scope(self._attention.name):
            projected_attention_keys = fflayer(
                inputs=attention_values, output_size=self._attention.attention_units,
                dropout_input_keep_prob=self.params["dropout_context_keep_prob"],
                activation=None, name="ff_att_keys")
        init_rnn_states = bridge(self._rnn_cells.state_size)
        if self._attention.attention_value_depth > 0:
            init_att_context = tf.zeros([tf.shape(attention_values)[0],
                                         self._attention.attention_value_depth], dtype=tf.float32)
        else:
            init_att_context = tf.zeros_like(attention_values[:, 0, :], dtype=tf.float32)
        init_cache = initialize_cache(
            decoding_states={"rnn_states": init_rnn_states,
                             "attention_context": init_att_context},
            attention_keys=projected_attention_keys,
            memory=attention_values,
            memory_bias=attention_bias)

        return init_cache

    def merge_top_features(self, decoder_output):
        """ Merges features of decoder top layers, as the input
        of softmax layer.
        Features to be merged are as follows:
            1. current decoder RNN state;
            2. current attention context.

        Args:
            decoder_output: An instance of `collections.namedtuple`
              whose element types are defined by `output_dtype`
              property.

        Returns: A instance of `tf.Tensor`, as the input of
          softmax layer.
        """
        assert isinstance(decoder_output, self._DecoderOutputSpec)
        cur_decoder_hidden = decoder_output.cur_decoder_hidden
        prev_input = decoder_output.prev_input
        attention_context = decoder_output.attention_context

        logit_lstm = fflayer(cur_decoder_hidden, output_size=self.params["logits_dimension"],
                             dropout_input_keep_prob=self.params["dropout_hidden_keep_prob"],
                             activation=None, name="ff_logit_lstm")
        # TODO here to fit old version code
        # logit_prev = fflayer(prev_input, output_size=self.params["logits_dimension"],
        #                      dropout_input_keep_prob=self.params["dropout_embedding_keep_prob"],
        #                      activation=None, name="ff_logit_prev")
        logit_ctx = fflayer(attention_context, output_size=self.params["logits_dimension"],
                            dropout_input_keep_prob=self.params["dropout_hidden_keep_prob"],
                            activation=None, name="ff_logit_ctx")
        # merged_output = tf.tanh(logit_lstm + logit_prev + logit_ctx)
        merged_output = tf.tanh(logit_lstm + logit_ctx)
        return merged_output

    def step(self, decoder_input, cache):
        """ Decodes one step.

        Args:
            decoder_input: The decoder input for this timestep, an
              instance of `tf.Tensor`, [batch_size, dim_word].
            cache: A dict containing decoder RNN states at previous
              timestep, pre-projected attention keys, attention values
              and attention length.

        Returns: A tuple `(cur_decoder_outputs, cur_cache)` at this timestep.
          The `cur_decoder_outputs` must be an instance of `collections.namedtuple`
          whose element types are defined by `output_dtype` property. The
          `cur_cache` must have the same structure with `cache`.
        """
        # run RNN
        cell_output, cell_states = self._rnn_cells(
            tf.concat([decoder_input, cache["decoding_states"]["attention_context"]], axis=1),
            cache["decoding_states"]["rnn_states"])

        # compute attention using hidden1
        # [batch_size, n_timesteps_src]
        attention_scores, attention_context = self._attention.build(
            query=cell_output,
            memory=cache["memory"],
            memory_bias=cache["memory_bias"],
            cache=cache)

        outputs = self._DecoderOutputSpec(
            cur_decoder_hidden=cell_output,
            prev_input=decoder_input,
            attention_context=attention_context,
            attention_scores=attention_scores)

        cache["decoding_states"]["rnn_states"] = cell_states
        cache["decoding_states"]["attention_context"] = attention_context
        return outputs, cache


class SimpleDecoder(Decoder):
    """ A simple RNN Decoder that doesn't use attention over an input sequence.
    """

    def __init__(self,
                 params,
                 mode,
                 name=None,
                 verbose=True):
        """ Initializes decoder parameters.

        Args:
            params: A dictionary of parameters to construct the
              decoder architecture.
            mode: A mode.
            name: The name of this decoder.
            verbose: Print decoder parameters if set True.
        """
        super(SimpleDecoder, self).__init__(params, mode, name, verbose)

        # recurrent cells
        self.rnn_cells = get_multilayer_rnn_cells(**self.params['rnn_cell'])
        self._decoder_output_tuple_type = namedtuple(
            "DecoderState", "cur_decoder_hidden prev_input")

    @property
    def output_dtype(self):
        """ Returns a `collections.namedtuple`,
        the definition of decoder output types. """
        return self._decoder_output_tuple_type(
            cur_decoder_hidden=tf.float32,
            prev_input=tf.float32)

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this decoder. """
        return {
            "rnn_cell": {
                "cell_class": "LSTMCell",
                "cell_params": {
                    "num_units": 1024,
                },
                "dropout_input_keep_prob": 1.0,
                "dropout_state_keep_prob": 1.0,
                "num_layers": 1,
            },
            "dropout_hidden_keep_prob": 1.0,
            "dropout_embedding_keep_prob": 1.0,
            "logits_dimension": 512,
        }

    def prepare(self, encoder_output, bridge, helper):
        """ Prepares for `step()` function.
        Do:
            1. initialize decoder RNN states using `bridge`;

        Args:
            encoder_output: An instance of `collections.namedtuple`
              from `Encoder.encode()`.
            bridge: An instance of `Bridge` that initializes the
              decoder states.
            helper: An instance of `Feedback` that samples next
              symbols from logits.
        Returns: A dict containing decoder RNN states and will be
          passed to `step()` function.
        """
        init_rnn_states = bridge(self.rnn_cells.state_size)
        init_cache = initialize_cache(
            decoding_states=init_rnn_states)
        return init_cache

    def merge_top_features(self, decoder_states):
        """ Merges features of decoder top layers, as the input
        of softmax layer.
        Features to be merged are as follows:
            1. current decoder RNN state;
            2. previous predicted word.

        Args:
            decoder_output: An instance of `collections.namedtuple`
              whose element types are defined by `output_dtype`
              property.

        Returns: A instance of `tf.Tensor`, as the input of
          softmax layer.
        """
        cur_decoder_hidden = decoder_states.cur_decoder_hidden
        prev_input = decoder_states.prev_input
        logit_lstm = fflayer(cur_decoder_hidden, output_size=self.params["logits_dimension"],
                             dropout_input_keep_prob=self.params["dropout_hidden_keep_prob"],
                             activation=None, name="ff_logit_lstm")
        logit_prev = fflayer(prev_input, output_size=self.params["logits_dimension"],
                             dropout_input_keep_prob=self.params["dropout_embedding_keep_prob"],
                             activation=None, name="ff_logit_prev")
        merged_output = tf.tanh(logit_lstm + logit_prev)
        return merged_output

    def step(self, decoder_input, cache):
        """ Decodes one step.

        Args:
            decoder_input: The decoder input for this timestep, an
              instance of `tf.Tensor`, [batch_size, dim_word].
            cache: A dict containing decoder RNN states at previous
              timestep.

        Returns: A tuple `(cur_decoder_outputs, cur_cache)` at this timestep.
          The `cur_decoder_outputs` must be an instance of `collections.namedtuple`
          whose element types are defined by `output_dtype` property. The
          `cur_cache` must have the same structure with `cache`.
        """
        # rnn layers
        cell_output, cell_states = self.rnn_cells(decoder_input, cache["decoding_states"])

        outputs = self._decoder_output_tuple_type(
            cur_decoder_hidden=cell_output,
            prev_input=decoder_input)
        cache["decoding_states"] = cell_states
        return outputs, cache
