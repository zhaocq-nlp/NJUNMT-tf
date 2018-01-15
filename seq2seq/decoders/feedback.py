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

import six
from abc import ABCMeta, abstractmethod
import tensorflow as tf
from seq2seq.utils.algebra_ops import advanced_log_softmax


def _unstack_ta(inp):
    return tf.TensorArray(
        dtype=inp.dtype, size=tf.shape(inp)[0],
        element_shape=inp.get_shape()[1:]).unstack(inp)


def _transpose_batch_time(x):
    """Transpose the batch and time dimensions of a Tensor.

    Retains as much of the static shape information as possible.

    Args:
      x: A tensor of rank 2 or higher.

    Returns:
      x transposed along the first two dimensions.

    Raises:
      ValueError: if `x` is rank 1 or lower.
    """
    x_static_shape = x.get_shape()
    if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
        raise ValueError(
            "Expected input tensor %s to have rank at least 2, but saw shape: %s" %
            (x, x_static_shape))
    x_rank = tf.rank(x)
    x_t = tf.transpose(
        x, tf.concat(
            ([1, 0], tf.range(2, x_rank)), axis=0))
    x_t.set_shape(
        tf.TensorShape([
            x_static_shape[1].value, x_static_shape[0].value
        ]).concatenate(x_static_shape[2:]))
    return x_t


@six.add_metaclass(ABCMeta)
class Feedback:
    def __init__(self, target_embedding_table, vocab, max_sequence_length):
        """

        :param target_embedding_table: target side embedding table
        :param sequence_length: (SOS, x, x, ...) not include EOS
        """
        self._target_embedding_table = target_embedding_table
        self.vocab = vocab
        self.max_seq_len = tf.convert_to_tensor(
            max_sequence_length, name="max_sequence_length")

    @property
    def dim_target_embedding(self):
        return self._target_embedding_table.get_shape().as_list()[-1]

    @property
    def target_embedding_table(self):
        return self._target_embedding_table.embedding_table

    @abstractmethod
    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def sample(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def next_inputs(self, *args, **kwargs):
        raise NotImplementedError


class TrainingFeedback(Feedback):
    def __init__(self, target_embedding_table, vocab, sequence, sequence_length):
        """
        contructor
        :param target_embedding_table: [target_vocab_size, dim_emb_trg]
        :param sequence: [batch_size, max_sequence_length], without sos id
        :param sequence_length: [batch_size]
        """
        super(TrainingFeedback, self).__init__(target_embedding_table, vocab, sequence_length)
        tf.logging.info("using TraningFeedBack......")
        sequence = tf.convert_to_tensor(sequence)
        sequence = _transpose_batch_time(self._target_embedding_table.embed_words(sequence))
        self._sequence_tas = _unstack_ta(sequence)

    def initialize(self):
        """
        initialize finished flags and first input with sos
        :return:
        """
        finished = tf.equal(0, self.max_seq_len)
        first_input_embs = self._target_embedding_table.embed_words(
            tf.tile([self.vocab.sos_id], [tf.shape(self.max_seq_len)[0]]))
        return finished, first_input_embs

    def sample(self, logits):
        """
        greedy sample
        :param logits:
        :return:
        """
        sample_ids = tf.cast(
            tf.argmax(logits, axis=-1), tf.int32)
        return sample_ids

    def next_inputs(self, time, sample_ids):
        """
        the input at "next time", is the output at "time"
        :param time: from 0 to ...
            the input at time0 is sos, the output at time0 is the input at time1...
        :param sample_ids: from sample(logits), [batch_size, ]
        :return:
        """
        next_time = time + 1
        finished = tf.greater_equal(next_time, self.max_seq_len)
        return finished, self._sequence_tas.read(time)


class BeamFeedback(Feedback):
    def __init__(self, target_embedding_table, vocab, max_seq_len, batch_size, beam_size):
        super(BeamFeedback, self).__init__(target_embedding_table, vocab, max_seq_len)
        tf.logging.info("using BeamFeedback with beam size=%d......" % (beam_size))
        self.batch_size = batch_size
        self.beam_size = beam_size

    def initialize(self):
        finished = tf.equal(0, self.max_seq_len)
        # [beam_size, ]
        finished = tf.tile([finished], [self.beam_size * self.batch_size])
        # [beam_size, target_emb_dim]
        inputs = self._target_embedding_table.embed_words(
            tf.tile([self.vocab.sos_id], [self.beam_size * self.batch_size]))
        return finished, inputs

    def sample(self, logits, log_probs, finished, lengths, time):
        """

        :param logits: [batch_size * beam_size, target_vocab_size]
        :param beam_state: .log_probs:  [batch_size * beam_size, ]
                           .finished: [batch_size * beam_size, ]
        :param time:
        :return:
        """
        # [batch_size * beam_size,]
        prev_finished_float = tf.to_float(finished)
        # [batch_size * beam_size, ]
        prev_log_probs = log_probs
        # [batch_size * beam_size, target_vocab_size]
        probs = advanced_log_softmax(logits)  # negative

        # mask the finished beam with only one entrance (target_eos_id)
        pseudo_float_min = (tf.reduce_sum(probs) + tf.reduce_sum(prev_log_probs)) * 10000.
        # [target_vocab_size]: [float_min, float_min, float_min, ...,  0]
        float_min_mask_vector = tf.one_hot([self.vocab.eos_id], self.vocab.vocab_size,
                                           on_value=0., off_value=pseudo_float_min, dtype=tf.float32)
        # [batch_size * beam_size, target_vocab_size]: outer product
        probs_mask_matrix = tf.tile(float_min_mask_vector, [self.beam_size * self.batch_size, 1])
        probs_mask_matrix *= tf.expand_dims(prev_finished_float, 1)
        probs = probs * tf.expand_dims(1. - prev_finished_float, 1) + probs_mask_matrix

        # [batch_size * beam_size, target_vocab_size]
        # compute new log_probs
        # log_probs = tf.expand_dims(1. - prev_finished_float, 1) * probs \
        #             + tf.expand_dims(prev_log_probs, 1)
        log_probs = probs + tf.expand_dims(prev_log_probs, 1)

        # flatten
        # [batch_size, beam_size * target_vocab_size]
        log_probs = tf.reshape(tf.reshape(log_probs, [-1]),
                               [self.batch_size, -1])

        log_probs_flat = tf.cond(
            tf.convert_to_tensor(time) > 0, lambda: log_probs,  # time > 0: all
            lambda: tf.slice(log_probs, [0, 0],
                             [-1, self.vocab.vocab_size]))  # time = 0: first logits in each batch

        # [batch_size, beam_size] will restore top live_k
        next_log_probs, word_ids = tf.nn.top_k(log_probs_flat, k=self.beam_size)

        # flatten: [batch_size * beam_size,]
        next_log_probs = tf.reshape(next_log_probs, [-1])
        word_ids = tf.reshape(word_ids, [-1])

        # indicating word ids in each batch sample
        # [batch_size, beam_size]
        sample_ids = tf.mod(word_ids, self.vocab.vocab_size)
        # beam ids should be adjusted according to batch_size
        # [beam_size, batch_size]: [[0, 1, 2,...], [0, 1, 2,...]]
        beam_add = tf.tile([tf.range(self.batch_size)],
                           [self.beam_size, 1]) * self.beam_size
        #  transpose beam_add: [[0, 0, ...], [1, 1,...], [batch_size,...] ]
        # [batch_size * beam_size, ]
        beam_ids = tf.div(word_ids, self.vocab.vocab_size) \
                   + tf.reshape(tf.transpose(beam_add), [-1])

        lengths = lengths + 1 - tf.to_int32(finished)

        return sample_ids, beam_ids, next_log_probs, tf.gather(lengths, beam_ids)

    def next_inputs(self, time, sample_ids):
        next_time = time + 1
        finished = tf.logical_or((next_time >= self.max_seq_len),
                                 tf.equal(self.vocab.eos_id, sample_ids))

        return finished, self._target_embedding_table.embed_words(sample_ids)
