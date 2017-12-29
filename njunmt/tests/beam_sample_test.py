from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from abc import ABCMeta, abstractmethod
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
import numpy

from njunmt.utils.algebra_ops import advanced_log_softmax
from njunmt.utils.beam_search import finished_beam_one_entry_bias
from njunmt.utils.beam_search import expand_to_beam_size
from njunmt.utils.beam_search import compute_batch_indices
from njunmt.utils.beam_search import gather_states
import tensorflow as tf

eos_id = 39
vocab_size = 40
alpha = 0.6
beam_size = 4
batch_size = 5


def sample_symbols_new(logits, log_probs, finished, lengths, time):
    """
    :param logits: [batch_size * beam_size, target_vocab_size]
    :param log_probs: [batch_size * beam_size, ]
    :param finished: [batch_size * beam_size, ]
    :param lengths: decoding length [batch_size * beam_size, ]
    :param time:
    :return:
    """

    # [batch_size * beam_size,]
    prev_finished_float = math_ops.to_float(finished)
    # [batch_size * beam_size, ]
    prev_log_probs = log_probs
    # [batch_size * beam_size, target_vocab_size]
    probs = advanced_log_softmax(logits)  # negative

    # mask the finished beam except only one entrance (target_eos_id)
    #   [target_vocab_size, ]: [float_min, float_min, float_min, ..., 0]
    #   this forces the beam with EOS continue to generate EOS
    finished_beam_bias = finished_beam_one_entry_bias(
        on_entry=eos_id, num_entries=vocab_size)
    # [batch_size * beam_size, target_vocab_size]: outer product
    finished_beam_bias = expand_to_beam_size(
        finished_beam_bias, beam_size * batch_size, axis=0)
    finished_beam_bias *= array_ops.expand_dims(prev_finished_float, 1)
    # compute new probs, with finished flags & mask
    probs = probs * array_ops.expand_dims(1. - prev_finished_float, 1) + finished_beam_bias

    # [batch_size * beam_size, target_vocab_size]
    # compute new log_probs
    log_probs = probs + array_ops.expand_dims(prev_log_probs, 1)
    # new decoding length: [batch_size * beam_size]
    lengths = lengths + 1 - math_ops.to_int32(finished)
    # compute beam score
    #  length_penalty: [batch_size * beam_size,]
    length_penalty = math_ops.pow(
        ((5.0 + math_ops.to_float(lengths)) / 6.0), -alpha)
    scores = log_probs * array_ops.expand_dims(length_penalty, axis=1)

    # flatten
    # [batch_size, beam_size * target_vocab_size]
    scores = array_ops.reshape(array_ops.reshape(scores, [-1]),
                               [batch_size, -1])
    ret_log_probs = array_ops.reshape(array_ops.reshape(log_probs, [-1]),
                                      [batch_size, -1])

    scores_flat = control_flow_ops.cond(
        ops.convert_to_tensor(time) > 0, lambda: scores,  # time > 0: all
        lambda: array_ops.slice(scores, [0, 0],
                                [-1, vocab_size]))  # time = 0: first logits in each batch

    # [batch_size, beam_size] will restore top live_k
    sample_scores, sample_ids = nn_ops.top_k(scores_flat, k=beam_size)
    ret_sample_ids = array_ops.reshape(sample_ids, [-1])
    # flatten: [batch_size * beam_size,]
    sample_ids = array_ops.reshape(sample_ids, [-1])
    # because we do topk to scores with dim:[batch, beam * vocab]
    #   we need to cover the true word ids
    word_ids = math_ops.mod(sample_ids, vocab_size)

    # beam ids should be adjusted according to batch_size
    #  batch_pos, [batch_size, beam_size]: [[0, 0, ...], [1, 1,...], [batch_size,...] ]
    batch_pos = compute_batch_indices(batch_size, beam_size)

    # compute new beam_ids, [batch_size * beam_size, ]
    beam_ids = math_ops.div(sample_ids, vocab_size) \
               + array_ops.reshape(batch_pos * beam_size, [-1])

    # we need to recover log_probs from score
    # flatten sample_scores: [batch_size * beam_size,]
    sample_scores_flatten = array_ops.reshape(sample_scores, [-1])
    # gather each length penalty
    length_penalty = gather_states(length_penalty, beam_ids)
    # recover log probabilities
    next_log_probs = sample_scores_flatten / length_penalty
    # gather states according to beam_ids
    next_lengths = gather_states(lengths, beam_ids)

    # [batch_size * beam_size * vocab_size, ]
    log_probs_flat = array_ops.reshape(log_probs, [-1])
    log_probs_index = array_ops.reshape(batch_pos, [-1]) * beam_size * vocab_size + sample_ids
    next_log_probs = array_ops.gather(log_probs_flat, log_probs_index)

    return word_ids, beam_ids, next_log_probs, next_lengths, ret_log_probs, ret_sample_ids, length_penalty


class BeamTest(tf.test.TestCase):
    def test_method1(self):
        logits = numpy.random.random(size=(beam_size * batch_size, vocab_size))
        log_probs = -numpy.random.random(size=(beam_size * batch_size))
        finished = numpy.array([0] * (batch_size * beam_size), dtype=bool)
        lengths = numpy.array([10] * (batch_size * beam_size), dtype=int)
        time = 10
        word_ids, beam_ids, next_log_probs, next_lengths, ret_log_probs, ret_sample_ids, length_penalty \
            = sample_symbols_new(tf.convert_to_tensor(logits, dtype=tf.float32),
                                 tf.convert_to_tensor(log_probs, dtype=tf.float32),
                                 tf.convert_to_tensor(finished, dtype=tf.bool),
                                 tf.convert_to_tensor(lengths, dtype=tf.int32),
                                 time)
        batch_pos = array_ops.reshape(compute_batch_indices(batch_size, beam_size), [-1])

        with self.test_session() as sess:
            ret_log_probs = sess.run(ret_log_probs)
            ret_sample_ids = sess.run(ret_sample_ids)
            batch_pos = sess.run(batch_pos)
            next_log_probs = sess.run(next_log_probs)
            self.assertAllEqual(ret_log_probs[batch_pos, ret_sample_ids], next_log_probs)

    def test_method2(self):
        logits = numpy.random.random(size=(beam_size * batch_size, vocab_size))
        log_probs = -numpy.random.random(size=(beam_size * batch_size))
        finished = numpy.array([0] * (batch_size * beam_size), dtype=bool)
        for i in range(beam_size * batch_size):
            if numpy.random.random() < 0.4:
                finished[i] = True
        lengths = numpy.array([10] * (batch_size * beam_size), dtype=int)
        for i in range(beam_size * batch_size):
            if numpy.random.random() < 0.4:
                finished[i] = True
                lengths[i] = numpy.random.randint(5, 10)
        time = 1
        word_ids, beam_ids, next_log_probs, next_lengths, ret_log_probs, ret_sample_ids, length_penalty \
            = sample_symbols_new(tf.convert_to_tensor(logits, dtype=tf.float32),
                                 tf.convert_to_tensor(log_probs, dtype=tf.float32),
                                 tf.convert_to_tensor(finished, dtype=tf.bool),
                                 tf.convert_to_tensor(lengths, dtype=tf.int32),
                                 time)
        batch_pos = array_ops.reshape(compute_batch_indices(batch_size, beam_size), [-1])

        with self.test_session() as sess:
            ret_log_probs = sess.run(ret_log_probs)
            ret_sample_ids = sess.run(ret_sample_ids)
            batch_pos = sess.run(batch_pos)
            next_log_probs = sess.run(next_log_probs)
            self.assertAllEqual(ret_log_probs[batch_pos, ret_sample_ids], next_log_probs)


if __name__ == "__main__":
    tf.test.main()
