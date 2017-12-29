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
""" Decoder helpers for sampling next symbols according to logits. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from abc import ABCMeta, abstractmethod
import tensorflow as tf
from tensorflow.python.util import nest

from njunmt.utils.algebra_ops import advanced_log_softmax
from njunmt.utils.algebra_ops import advanced_softmax
from njunmt.utils.beam_search import finished_beam_one_entry_bias
from njunmt.utils.beam_search import expand_to_beam_size
from njunmt.utils.beam_search import compute_batch_indices
from njunmt.utils.beam_search import gather_states


def _unstack_ta(inp):
    return tf.TensorArray(
        clear_after_read=False,
        dtype=inp.dtype, size=tf.shape(inp)[0],
        element_shape=inp.get_shape()[1:]).unstack(inp)


def _transpose_batch_time(x):
    """Transpose the batch and time dimensions of a Tensor.

    Retains as much of the static shape information as possible.

    Args:
        x: A tensor of rank 2 or higher.

    Returns: x transposed along the first two dimensions.

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
    """ Base class for decoder helpers. """

    def __init__(self, vocab, maximum_labels_length):
        """ Initializes the base feedback.

        Args:
            vocab: A `Vocab` object.
            maximum_labels_length: The maximum sequence length
              that decoder generates.
        """
        self._beam_size = 1
        self._vocab = vocab
        self._maximum_labels_length = tf.convert_to_tensor(
            maximum_labels_length, name="max_sequence_length")

    @property
    def beam_size(self):
        """ Returns the beam width. """
        return self._beam_size

    @property
    def vocab(self):
        """Returns the vocabulary. """
        return self._vocab

    @abstractmethod
    def init_symbols(self, *args, **kwargs):
        """ Returns the first input symbols of the decoder. """
        raise NotImplementedError

    @abstractmethod
    def sample_symbols(self, *args, **kwargs):
        """ Returns the sample symbols at current status."""
        raise NotImplementedError

    @abstractmethod
    def next_symbols(self, *args, **kwargs):
        """ Returns the next decoder input symbols. """
        raise NotImplementedError


class ScheduleSampleFeedback(Feedback):
    """ Define a helper class for training with schedule sampling. """

    def __init__(self, vocab, label_ids, label_length):
        """ Initializes the training feedback.

        Args:
            vocab: A `Vocab` object.
            label_ids: The gold labels Tensor, with shape [batch_size, maximum_labels_length].
            label_length: The length Tensor of `label_ids`, with shape [batch_size, ]
        """
        super(TrainingFeedback, self).__init__(vocab, label_length)
        label_ids = tf.convert_to_tensor(label_ids)  # [batch, len]
        self.label_ids = label_ids  # for transformer

    def init_symbols(self):
        # TODO
        raise NotImplementedError

    def sample_symbols(self, logits):
        # TODO
        raise NotImplementedError

    def next_symbols(self, time, sample_ids):
        # TODO
        raise NotImplementedError


class TrainingFeedback(Feedback):
    """ Define a helper class for training. """

    def __init__(self, vocab, label_ids, label_length):
        """ Initializes the training feedback.

        Args:
            vocab: A `Vocab` object.
            label_ids: The gold labels Tensor, with shape [batch_size, maximum_labels_length].
            label_length: The length Tensor of `label_ids`, with shape [batch_size, ]
        """
        super(TrainingFeedback, self).__init__(vocab, label_length)
        label_ids = tf.convert_to_tensor(label_ids)  # [batch, len]
        self.label_ids = label_ids  # for transformer

    def init_symbols(self):
        """ Returns a tuple `(init_finished_flags, init_input_symbols)`, where
        `init_finished_flags` contains all False values and `init_input_symbols`
        contains the index of start of sentence symbol. Both of two tensors have
        shape [batch_size, ]
        """
        transposed_label_ids = _transpose_batch_time(self.label_ids)  # [len, batch]
        self._label_sequence_tas = _unstack_ta(transposed_label_ids)
        finished = tf.equal(0, self._maximum_labels_length)
        first_input_embs = tf.tile([self._vocab.sos_id], [tf.shape(self._maximum_labels_length)[0]])
        return finished, first_input_embs

    def sample_symbols(self, logits):
        """ Samples symbols and returns it, a Tensor with shape [batch_size, ] """
        raise NotImplementedError(
            "There is no need to call this function in TrainingFeedback.")

    def next_symbols(self, time, sample_ids):
        """ Returns the output at `time`, also known as the
        input at `time`+1.

        Args:
            time: A int32 Scalar, the current time.
            sample_ids: A Tensor with shape [batch_size, ], returned by
              `sample_symbols()`.

        Returns: A tuple `(finished, next_symbols)`, where `finished` indicates
          whether each sequence is finished, and `next_symbols` is the next input
          Tensor with shape [batch_size, ]
        """
        _ = sample_ids
        next_time = time + 1
        finished = tf.greater_equal(next_time, self._maximum_labels_length)
        return finished, self._label_sequence_tas.read(time)


class BeamFeedback(Feedback):
    """ Define a helper class for inference with beam search. """

    def __init__(self, vocab, maximum_labels_length,
                 batch_size, beam_size, alpha=None,
                 ensemble_weight=None):
        """ Initializes the feedback for beam search.

        Args:
            vocab: A `Vocab` object.
            maximum_labels_length: A python integer, the maximum sequence
              length that decoder generates.
            batch_size: The batch size.
            beam_size: The beam width.
            alpha: The length penalty rate. Length penalty is given by
                    (5+len(decode)/6) ^ -\alpha.
              Refer to https://arxiv.org/abs/1609.08144.
            ensemble_weight: None or a list of floats to average the log
              probabilities from many models..
        """
        super(BeamFeedback, self).__init__(vocab, maximum_labels_length)
        self._batch_size = batch_size
        self._beam_size = beam_size
        self._alpha = alpha
        if alpha is None or alpha < 0.0:
            self._alpha = 0.0
        self._ensemble_weights = ensemble_weight

    def init_symbols(self):
        """ Returns a tuple `(init_finished_flags, init_input_symbols)`, where
        `init_finished_flags` contains all False values and `init_input_symbols`
        contains the index of start of sentence symbol. Both of two tensors have
        shape [batch_size, ]
        """
        finished = tf.equal(0, self._maximum_labels_length)
        # [batch_size * beam_size, ]
        finished = tf.tile([finished], [self._beam_size * self._batch_size])
        # [beam_size, target_emb_dim]
        inputs = tf.tile([self._vocab.sos_id], [self._beam_size * self._batch_size])
        return finished, inputs

    def _compute_log_probs(self, logits):
        """ Computes log probabilities.

        Here, `num_samples` == `beam_size` * `batch_size`.

        Args:
            logits: The logits Tensor with shape [num_samples, vocab_size],
              or a list of logits Tensors.

        Returns: The log probability Tensor with shape [num_samples, vocab_size].

        """
        logits = nest.flatten(logits)
        if len(logits) == 1:
            probs = advanced_log_softmax(logits[0])  # negative
        else:
            assert len(logits) == len(self._ensemble_weights), (
                "ensemble weights must have the same length with logits")
            dim_vocab = logits[0].get_shape().as_list()[-1]
            # [1, batch_size * beam_size * vocab_target]
            probs = nest.map_structure(
                lambda x: tf.expand_dims(
                    tf.reshape(advanced_softmax(x), shape=[-1]), axis=0),
                logits)
            # [num_models, xxx]
            probs = tf.concat(probs, axis=0)
            # [1, num_models]
            weights = tf.expand_dims(
                tf.convert_to_tensor(self._ensemble_weights, dtype=tf.float32),
                axis=0)
            probs = tf.matmul(weights, probs)
            probs = tf.log(tf.reshape(probs, [-1, dim_vocab]))
        return probs

    def sample_symbols(self, logits, log_probs, finished, lengths, time):
        """ Samples symbols and returns it.

        Args:
            logits: The logits Tensor with shape [beam_size * batch_size, vocab_size],
              or a list of logits Tensors.
            log_probs: Accumulated log probabilities, a float32 Tensor with shape
              [beam_size * batch_size, ].
            finished: Finished flag of each beam in each batch, a bool Tensor with
              shape [beam_size * batch_size, ].
            lengths: The length of each beam in each batch, a int32 Tensor with
              shape [beam_size * batch_size, ].
            time: A int32 Scalar, the current time.

        Returns: A tuple `(word_ids, beam_ids, next_log_probs, next_lengths)`, where
          `words_ids` is the ids of sampled word symbols; `beam_ids` indicates the index
          of beam which the symbol at the position is from; `next_log_probs` is the accumulated
          log probabilities of each beam; `next_lengths` is the decoding lengths of
          each beam.
          All of the Tensors have shape [batch_size * beam_size, ].
        """
        # [batch_size * beam_size,]
        prev_finished_float = tf.to_float(finished)
        # [batch_size * beam_size, ]
        prev_log_probs = log_probs
        # [batch_size * beam_size, target_vocab_size]
        probs = self._compute_log_probs(logits)

        # mask the finished beam except only one entrance (target_eos_id)
        #   [target_vocab_size, ]: [float_min, float_min, float_min, ..., 0]
        #   this forces the beam with EOS continue to generate EOS
        finished_beam_bias = finished_beam_one_entry_bias(
            on_entry=self._vocab.eos_id, num_entries=self._vocab.vocab_size)
        # [batch_size * beam_size, target_vocab_size]: outer product
        finished_beam_bias = expand_to_beam_size(
            finished_beam_bias, self._beam_size * self._batch_size, axis=0)
        finished_beam_bias *= tf.expand_dims(prev_finished_float, 1)
        # compute new probs, with finished flags & mask
        probs = probs * tf.expand_dims(1. - prev_finished_float, 1) + finished_beam_bias

        # [batch_size * beam_size, target_vocab_size]
        # compute new log_probs
        log_probs = probs + tf.expand_dims(prev_log_probs, 1)
        # new decoding length: [batch_size * beam_size]
        lengths = lengths + 1 - tf.to_int32(finished)
        # compute beam score
        #  length_penalty: [batch_size * beam_size,]
        length_penalty = tf.pow(
            ((5.0 + tf.to_float(lengths)) / 6.0), -self._alpha)
        scores = log_probs * tf.expand_dims(length_penalty, axis=1)

        # flatten: [batch_size, beam_size * target_vocab_size]
        scores = tf.reshape(tf.reshape(scores, [-1]),
                            [self._batch_size, -1])
        scores_flat = tf.cond(
            tf.convert_to_tensor(time) > 0, lambda: scores,  # time > 0: all
            lambda: tf.slice(scores, [0, 0],
                             [-1, self._vocab.vocab_size]))  # time = 0: first logits in each batch

        # [batch_size, beam_size] will restore top live_k
        sample_scores, sample_ids = tf.nn.top_k(scores_flat, k=self._beam_size)
        # flatten: [batch_size * beam_size,]
        sample_ids = tf.reshape(sample_ids, [-1])

        # because we do topk to scores with dim:[batch, beam * vocab]
        #   we need to cover the true word ids
        word_ids = tf.mod(sample_ids, self._vocab.vocab_size)

        # find beam_ids, indicating the current position is from which beam
        #  batch_pos, [batch_size, beam_size]: [[0, 0, ...], [1, 1,...], ..., [batch_size,...] ]
        batch_pos = compute_batch_indices(self._batch_size, self._beam_size)
        #  beam_base_pos: [batch_size * beam_size,]: [0, 0, ..., beam, beam,..., 2beam, 2beam, ...]
        beam_base_pos = tf.reshape(batch_pos * self._beam_size, [-1])
        # compute new beam_ids, [batch_size * beam_size, ]
        beam_ids = tf.div(sample_ids, self._vocab.vocab_size) + beam_base_pos

        # gather states according to beam_ids
        next_lengths = gather_states(lengths, beam_ids)

        # we need to recover log_probs according to scores's topk ids
        # [batch_size * beam_size * vocab_size, ]
        log_probs_flat = tf.reshape(log_probs, [-1])
        log_probs_index = beam_base_pos * self._vocab.vocab_size + sample_ids
        next_log_probs = tf.gather(log_probs_flat, log_probs_index)

        return word_ids, beam_ids, next_log_probs, next_lengths

    def next_symbols(self, time, sample_ids):
        """ Returns the output at `time`, also known as the
        input at `time`+1.

        Args:
            time: A int32 Scalar, the current time.
            sample_ids: A Tensor with shape [batch_size, ], returned by
              `sample_symbols()`.

        Returns: A tuple `(finished, next_symbols)`, where `finished` indicates
          whether each sequence is finished, and `next_symbols` is the next input
          Tensor with shape [batch_size * beam_size, ]
        """
        next_time = time + 1
        finished = tf.logical_or((next_time >= self._maximum_labels_length),
                                 tf.equal(self._vocab.eos_id, sample_ids))

        return finished, sample_ids


if __name__ == "__main__":
    a = tf.convert_to_tensor([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
    a_t = _transpose_batch_time(a)
    with tf.Session() as sess:
        print(sess.run(a_t))
