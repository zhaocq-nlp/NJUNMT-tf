# Copyright 2017 ZhaoChengqi, zhaocq@nlp.nju.edu.cn, Natural Language Processing Group, Nanjing University.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import tensorflow as tf
from tensorflow.python.util import nest

from seq2seq.decoders.decoder import Decoder
from seq2seq.utils.rnn_cell_utils import get_condr_rnn_cell
from seq2seq.utils.rnn_cell_utils import get_multilayer_rnn_cells
from seq2seq.utils.algebra_ops import fflayer


def _gather_state(state, idx):
    def _gather(s):
        return tf.gather(s, idx)

    return nest.pack_sequence_as(state,
                                 nest.map_structure(
                                     _gather,
                                     nest.flatten(state)))


def _stack_beam_size(x, batch_size, beam_size):
    """
    for batch beam search
    stack beam size for the tensor
    :param x:
    :param batch_size:
    :param beam_size:
    :return:
    """
    if x.get_shape().ndims == 3:
        # tile: [batch_size, max_len_seq * beam_size, num_units]
        #    reshape: [batch_size * beam_size, max_len_seq, num_units]
        final_shape = [beam_size * batch_size, -1, x.get_shape().as_list()[-1]]
        return tf.reshape(tf.tile(x, [1, beam_size, 1]), final_shape)
    elif x.get_shape().ndims == 1:
        return tf.reshape(
            tf.transpose(tf.tile([x], [beam_size, 1])), [-1])
    else:
        raise NotImplementedError("Not implemented the capability for ndims=%d" % x.get_shape().ndims)


class CondAttentionDecoder(Decoder):
    """
    An RNN Decoder that uses attention over an input sequence.
    conditional LSTM or GRU as base layer
    the following layers use Stacked rGRU represented in
        arXiv:1707.07631 Deep Architectures for Neural Machine Translation
    """

    def __init__(self, params, mode,
                 vocab_target_size):
        """
        conditional LSTM/GRU decoder
        :param params:
        :param mode: Modekeys.TRAIN / INFER / EVAL
        :param vocab_target_size: size of target vocabulary
        """
        super(CondAttentionDecoder, self).__init__(params, mode)

        # recurrent cells
        self.cond_rnn_cell, self.r_rnn_cells = get_condr_rnn_cell(**self.params['rnn_cell'])
        self.vocab_target_size = vocab_target_size
        # for attention
        self.attention_units = self.params["attention_units"]

    @staticmethod
    def default_params():
        return {
            "attention_units": 2048,
            "rnn_cell": {
                "cell_class": "LSTMCell",
                "cell_params": {
                    "num_units": 1024,
                },
                "dropout_input_keep_prob": 1.0,
                "dropout_state_keep_prob": 1.0,
                "num_layers": 1,
            },
            "dropout_context_keep_prob": 1.0,
            "dropout_hidden_keep_prob": 1.0,
            "dropout_logit_keep_prob": 1.0,
            "dropout_embedding_keep_prob": 1.0,
            "dim.logit": 512,
            "init_scale": 0.04
        }

    def _setup(self, encoder_output, bridge, helper):
        """ set up the decoder helpers for decoding

        :param encoder_output: encoder output named tuple
        :param bridge: `Bridge` sub instance
        :param helper: `FeedBack` sub instance
        :return:
        """
        self.helper = helper
        self.bridge = bridge
        self.att_values = encoder_output.attention_values
        self.att_values_lengths = encoder_output.attention_length
        self._batch_size = tf.shape(self.att_values_lengths)[0]

        # build output type tuples
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN \
                or self.mode == tf.contrib.learn.ModeKeys.EVAL:

            self.output_tuple_type = namedtuple("DecoderOutput", [
                "cur_decoder_hidden", "prev_input", "attention_context",
                "attention_scores"])
            self._output_size = self.output_tuple_type(
                cur_decoder_hidden=self.cond_rnn_cell.output_size,
                prev_input=helper.dim_target_embedding,
                attention_context=tf.shape(self.att_values)[-1],
                attention_scores=tf.shape(self.att_values)[1:-1])
            self._output_dtype = self.output_tuple_type(
                cur_decoder_hidden=tf.float32,
                prev_input=tf.float32,
                attention_context=tf.float32,
                attention_scores=tf.float32)
        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            self._beam_size = helper.beam_size
            self.output_tuple_type = namedtuple("DecoderOutput", [
                "logits", "log_probs",
                "predicted_ids", "beam_ids", "lengths",
                "attention_scores"])
            self._output_size = self.output_tuple_type(
                logits=self.vocab_target_size,
                log_probs=tf.TensorShape([]),
                predicted_ids=tf.TensorShape([]),
                beam_ids=tf.TensorShape([]),
                lengths=tf.TensorShape([]),
                attention_scores=tf.shape(self.att_values)[1:-1])
            self._output_dtype = self.output_tuple_type(
                logits=tf.float32,
                log_probs=tf.float32,
                predicted_ids=tf.int32,
                beam_ids=tf.int32,
                lengths=tf.int32,
                attention_scores=tf.float32)
        else:
            raise ValueError("Unrecognized mode key: %s" % self.mode)

    def initialize(self):
        """
        initialize the deocder states
        for TrainFeedBack -- finished is of [_batch_size] else [_batch_size * beam_size]
        :return:
        """
        self.precal_att_keys = fflayer(self.att_values, output_size=self.attention_units, activation=None,
                                       dropout_input_keep_prob=self.params["dropout_context_keep_prob"],
                                       name="ff_att_keys")
        finished, first_inputs = self.helper.initialize()
        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            # [batch_size * beam_size, max_len_seq, num_attention_units]
            self.precal_att_keys = _stack_beam_size(self.precal_att_keys, self._batch_size, self.beam_size)
            # [batch_size * beam_size, max_len_seq, num_values]
            self.att_values = _stack_beam_size(self.att_values, self._batch_size, self.beam_size)
            # [batch_size * beam_size, ]
            self.att_values_lengths = _stack_beam_size(self.att_values_lengths, self._batch_size, self.beam_size)

            # rnn_init_states = self.bridge(self.rnn_cells.state_size, self.att_values, self.att_values_lengths)
            rnn_init_states = self.bridge(self.r_rnn_cells.state_size, beam_size=self.beam_size)
            log_probs = tf.zeros([self.batch_size, ], dtype=tf.float32)
            lengths = tf.zeros([self.batch_size, ], dtype=tf.int32)
            return finished, first_inputs, rnn_init_states, (log_probs, finished, lengths)

        rnn_init_states = self.bridge(self.r_rnn_cells.state_size)
        # first_inputs: [batch_size, dim_target_emb]
        return finished, first_inputs, rnn_init_states, tuple()

    def compute_logit(self, decoder_output, scope=None):
        """
        compute logit according to decoder output
          to speed up training procedure
        :param decoder_output: DecoderOutputTuple for TRAIN/EVAL, tuple for INFER
        :param scope:
        :return:
        """
        if type(decoder_output) is tuple or type(decoder_output) is list:
            # [batch_size, dim]
            cell_output, inputs, attention_context = decoder_output
        elif type(decoder_output) is self.output_tuple_type:
            # [timesteps_trg, batch_size, dim]
            cell_output = decoder_output.cur_decoder_hidden
            inputs = decoder_output.prev_input
            attention_context = decoder_output.attention_context
        else:
            raise ValueError("Unrecognized decoder_output type: %s, "
                             "which should be tupe or DecoderOutput" % decoder_output)

        cur_scope = tf.get_variable_scope()
        var_scope_name = scope or cur_scope
        with tf.variable_scope(var_scope_name):
            logit_lstm = fflayer(cell_output, output_size=self.params["dim.logit"],
                                 activation=None,
                                 dropout_input_keep_prob=self.params["dropout_hidden_keep_prob"],
                                 name="ff_logit_lstm")
            logit_prev = fflayer(inputs, output_size=self.params["dim.logit"],
                                 activation=None,
                                 dropout_input_keep_prob=self.params["dropout_embedding_keep_prob"],
                                 name="ff_logit_prev")
            logit_ctx = fflayer(attention_context, output_size=self.params["dim.logit"],
                                activation=None,
                                dropout_input_keep_prob=self.params["dropout_hidden_keep_prob"],
                                name="ff_logit_ctx")

            logits = tf.tanh(logit_lstm + logit_prev + logit_ctx)
            logits = fflayer(logits, output_size=self.vocab_target_size,
                             activation=None, dropout_input_keep_prob=self.params["dropout_logit_keep_prob"],
                             name="ff_logit")
        return logits

    def finalize(self, final_outputs, final_state):
        return final_outputs, final_state

    def _attention_fn(self, query, name=None):
        """ BahdanauAttention fn
        :param query:
        :return:
        """
        query_trans = fflayer(query, output_size=self.attention_units, activation=None,
                              dropout_input_keep_prob=self.params["dropout_hidden_keep_prob"],
                              name="query_trans")

        # expandedd att_query: [batch_size, 1, num_units]
        # energres: [batch_size, max_len]
        with tf.variable_scope(name or "BahdanauAttention"):
            v_att = tf.get_variable("v_att", shape=[self.attention_units], dtype=tf.float32)
        energies = tf.reduce_sum(v_att * tf.tanh(
            self.precal_att_keys + tf.expand_dims(query_trans, 1)), [2])

        # Replace all scores for padded inputs with tf.float32.min
        num_scores = tf.shape(energies)[1]  # max length of keys
        scores_mask = tf.sequence_mask(
            lengths=tf.to_int32(self.att_values_lengths),
            maxlen=tf.to_int32(num_scores),
            dtype=tf.float32)

        energies = energies * scores_mask + ((1.0 - scores_mask) * (-1.e9))

        # Stabilize energies first and then exp
        energies = energies - tf.reduce_max(energies, axis=1, keep_dims=True)
        unnormalized_scores = tf.exp(energies) * scores_mask

        normalization = tf.reduce_sum(unnormalized_scores, axis=1, keep_dims=True)

        # Normalize the scores
        # [batch_size, 1]
        scores_normalized = unnormalized_scores / normalization

        # Calculate the weighted average of the attention inputs
        # according to the scores
        #   [batch_size, 1, 1] * [batch_size, max_len, dim_values]
        context = tf.expand_dims(scores_normalized, 2) * self.att_values
        #   [batch_size, dim_values]
        context = tf.reduce_sum(context, 1, name="context")
        context.set_shape([None, self.att_values.get_shape().as_list()[-1]])

        return scores_normalized, context

    def step(self, time_, inputs, decoder_state, aux_inputs):
        """
        decode one step
        :param time_: scalar, current decoding time, start from 0
        :param inputs: decdoer input, [batch_size, dim_word]
        :param decoder_state:
        :param aux_inputs: more inputs
        :return:
        """
        # layer0: get hidden1
        cell_output0, cell_state0 = self.cond_rnn_cell(inputs, decoder_state[0])

        # compute attention using hidden1
        # [batch_size, n_timesteps_src]
        attention_scores, attention_context = self._attention_fn(query=cell_output0)

        # hidden1's state is the hidden2 's initstate
        following_decoder_state = tuple([cell_state0] + list(decoder_state[1:]))
        cell_output, cell_states = self.r_rnn_cells(attention_context, following_decoder_state)
        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            log_probs, finished, lengths = aux_inputs
            logits = self.compute_logit((cell_output, inputs, attention_context))
            sample_ids, beam_ids, next_log_probs, next_lengths \
                = self.helper.sample(logits, log_probs, finished, lengths, time=time_)

            next_cell_states = _gather_state(cell_states, beam_ids)
            outputs = self.output_tuple_type(
                logits=logits,
                log_probs=next_log_probs,
                predicted_ids=sample_ids,
                beam_ids=beam_ids,
                lengths=next_lengths,
                attention_scores=attention_scores)

            next_finished, next_inputs = self.helper.next_inputs(time=time_,
                                                                 sample_ids=sample_ids)
            return outputs, next_finished, next_inputs, next_cell_states, \
                   (next_log_probs, next_finished, next_lengths)

        outputs = self.output_tuple_type(
            cur_decoder_hidden=cell_output,
            prev_input=inputs,
            attention_context=attention_context,
            attention_scores=attention_scores)

        finished, next_inputs = self.helper.next_inputs(time=time_,
                                                        sample_ids=None)
        # return outputs, cell_states, next_inputs, tuple(), finished
        return outputs, finished, next_inputs, cell_states, tuple()


class SimpleDecoder(Decoder):
    """
    An simple RNN Decoder
    """

    def __init__(self, params, mode,
                 vocab_target_size):
        """
        conditional LSTM/GRU decoder
        :param params:
        :param mode: Modekeys.TRAIN / INFER / EVAL
        :param vocab_target_size: size of target vocabulary
        """
        super(SimpleDecoder, self).__init__(params, mode)

        # recurrent cells
        self.cells = get_multilayer_rnn_cells(**self.params['rnn_cell'])
        self.vocab_target_size = vocab_target_size

    @staticmethod
    def default_params():
        return {
            "attention_units": 2048,
            "rnn_cell": {
                "cell_class": "LSTMCell",
                "cell_params": {
                    "num_units": 1024,
                },
                "dropout_input_keep_prob": 1.0,
                "dropout_state_keep_prob": 1.0,
                "num_layers": 1,
            },
            "dropout_context_keep_prob": 1.0,
            "dropout_hidden_keep_prob": 1.0,
            "dropout_logit_keep_prob": 1.0,
            "dropout_embedding_keep_prob": 1.0,
            "dim.logit": 512,
            "init_scale": 0.04
        }

    def _setup(self, encoder_output, bridge, helper):
        """ set up the decoder helpers for decoding

        :param encoder_output: encoder output named tuple
        :param bridge: `Bridge` sub instance
        :param helper: `FeedBack` sub instance
        :return:
        """
        self.helper = helper
        self.bridge = bridge
        self._batch_size = tf.shape(encoder_output.attention_length)[0]

        # build output type tuples
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN \
                or self.mode == tf.contrib.learn.ModeKeys.EVAL:

            self.output_tuple_type = namedtuple("DecoderOutput", [
                "cur_decoder_hidden", "prev_input"])
            self._output_size = self.output_tuple_type(
                cur_decoder_hidden=self.cells.output_size,
                prev_input=helper.dim_target_embedding)
            self._output_dtype = self.output_tuple_type(
                cur_decoder_hidden=tf.float32,
                prev_input=tf.float32)
        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            self._beam_size = helper.beam_size
            self.output_tuple_type = namedtuple("DecoderOutput", [
                "logits", "log_probs",
                "predicted_ids", "beam_ids", "lengths"])
            self._output_size = self.output_tuple_type(
                logits=self.vocab_target_size,
                log_probs=tf.TensorShape([]),
                predicted_ids=tf.TensorShape([]),
                beam_ids=tf.TensorShape([]),
                lengths=tf.TensorShape([]))
            self._output_dtype = self.output_tuple_type(
                logits=tf.float32,
                log_probs=tf.float32,
                predicted_ids=tf.int32,
                beam_ids=tf.int32,
                lengths=tf.int32)
        else:
            raise ValueError("Unrecognized mode key: %s" % self.mode)

    def initialize(self):
        """
        initialize the deocder states
        for TrainFeedBack -- finished is of [_batch_size] else [_batch_size * beam_size]
        :return:
        """
        finished, first_inputs = self.helper.initialize()
        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            # rnn_init_states = self.bridge(self.rnn_cells.state_size, self.att_values, self.att_values_lengths)
            rnn_init_states = self.bridge(self.cells.state_size, beam_size=self.beam_size)
            log_probs = tf.zeros([self.batch_size, ], dtype=tf.float32)
            lengths = tf.zeros([self.batch_size, ], dtype=tf.int32)
            return finished, first_inputs, rnn_init_states, (log_probs, finished, lengths)

        rnn_init_states = self.bridge(self.cells.state_size)
        # first_inputs: [batch_size, dim_target_emb]
        return finished, first_inputs, rnn_init_states, tuple()

    def compute_logit(self, decoder_output, scope=None):
        """
        compute logit according to decoder output
          to speed up training procedure
        :param decoder_output: DecoderOutputTuple for TRAIN/EVAL, tuple for INFER
        :param scope:
        :return:
        """
        if type(decoder_output) is tuple or type(decoder_output) is list:
            # [batch_size, dim]
            cell_output, inputs = decoder_output
        elif type(decoder_output) is self.output_tuple_type:
            # [timesteps_trg, batch_size, dim]
            cell_output = decoder_output.cur_decoder_hidden
            inputs = decoder_output.prev_input
        else:
            raise ValueError("Unrecognized decoder_output type: %s, "
                             "which should be tupe or DecoderOutput" % decoder_output)

        cur_scope = tf.get_variable_scope()
        var_scope_name = scope or cur_scope
        with tf.variable_scope(var_scope_name):
            logit_lstm = fflayer(cell_output, output_size=self.params["dim.logit"],
                                 activation=None,
                                 dropout_input_keep_prob=self.params["dropout_hidden_keep_prob"],
                                 name="ff_logit_lstm")
            logit_prev = fflayer(inputs, output_size=self.params["dim.logit"],
                                 activation=None,
                                 dropout_input_keep_prob=self.params["dropout_embedding_keep_prob"],
                                 name="ff_logit_prev")

            logits = tf.tanh(logit_lstm + logit_prev)
            logits = fflayer(logits, output_size=self.vocab_target_size,
                             activation=None, dropout_input_keep_prob=self.params["dropout_logit_keep_prob"],
                             name="ff_logit")
        return logits

    def finalize(self, final_outputs, final_state):
        return final_outputs, final_state

    def step(self, time_, inputs, decoder_state, aux_inputs):
        """
        decode one step
        :param time_: scalar, current decoding time, start from 0
        :param inputs: decdoer input, [batch_size, dim_word]
        :param decoder_state:
        :param aux_inputs: more inputs
        :return:
        """
        # layer0: get hidden
        cell_output, cell_states = self.cells(inputs, decoder_state)

        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            log_probs, finished, lengths = aux_inputs
            logits = self.compute_logit((cell_output, inputs))
            sample_ids, beam_ids, next_log_probs, next_lengths \
                = self.helper.sample(logits, log_probs, finished, lengths, time=time_)

            next_cell_states = _gather_state(cell_states, beam_ids)
            outputs = self.output_tuple_type(
                logits=logits,
                log_probs=next_log_probs,
                predicted_ids=sample_ids,
                beam_ids=beam_ids,
                lengths=next_lengths)

            next_finished, next_inputs = self.helper.next_inputs(time=time_,
                                                                 sample_ids=sample_ids)
            return outputs, next_finished, next_inputs, next_cell_states, \
                   (next_log_probs, next_finished, next_lengths)

        outputs = self.output_tuple_type(
            cur_decoder_hidden=cell_output,
            prev_input=inputs)

        finished, next_inputs = self.helper.next_inputs(time=time_,
                                                        sample_ids=None)
        # return outputs, cell_states, next_inputs, tuple(), finished
        return outputs, finished, next_inputs, cell_states, tuple()
