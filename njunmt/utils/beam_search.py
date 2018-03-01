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
""" Define common functions for beam search. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from tensorflow.python.util import nest
import tensorflow as tf


class BeamSearchStateSpec(
    namedtuple(
        "BeamSearchStat", "log_probs beam_ids")):
    """ A class wrapper for namedtuple.  """

    @staticmethod
    def dtypes():
        """ Returns the types of this namedtuple. """
        return BeamSearchStateSpec(
            log_probs=tf.float32,
            beam_ids=tf.int32)


def stack_beam_size(tensors, beam_size):
    """ Stacks the tensors `beam_size` times at specific dimension.

    Args:
        tensors: A Tensor of a list/tuple/dict of Tensors. For each Tensor, the first
          dimension must be batch_size, otherwise, unknow errors may occur.
        beam_size: A python integer, the beam width.

    Returns: A Tensor or a list/tuple of Tensors with the same structure
      as `tensors`.

    Raises:
        AssertionError: if the shape of tensor does not match
          [batch_size, 1, 1, timesteps] when tensor.ndims == 4.
        NotImplementedError: if tensor.ndims > 4.
    """

    def _stack(x):
        assert isinstance(x, tf.Tensor)
        batch_size = tf.shape(x)[0]
        x_ndims = x.get_shape().ndims
        last_dim = x.get_shape().as_list()[-1]
        if last_dim is None:
            last_dim = tf.shape(x)[-1]
        if x_ndims == 3:
            final_shape = [beam_size * batch_size, -1, last_dim]
            return tf.reshape(tf.tile(x, [1, beam_size, 1]), final_shape)
        elif x_ndims == 2:
            final_shape = [batch_size * beam_size, last_dim]
            return tf.reshape(tf.tile(x, [1, beam_size]), final_shape)
        elif x_ndims == 1:
            return tf.reshape(
                tf.transpose(tf.tile([x], [beam_size, 1])), [-1])
        elif x_ndims == 4:
            assert x.get_shape().as_list()[1] == x.get_shape().as_list()[2] == 1, (
                "this only matches the bias tensor with shape [batch_size, 1, 1, timesteps]")
            return tf.expand_dims(
                _stack(tf.squeeze(x, axis=1)), axis=1)
        else:
            raise NotImplementedError("Not implemented the capability for ndims={}".format(x_ndims))

    return nest.pack_sequence_as(
        tensors,
        nest.map_structure(
            _stack, nest.flatten(tensors)))


def gather_states(states, beam_ids):
    """ Gathers states according to beam ids.

    Args:
        states: A Tensor of a list/tuple/dict of Tensors. For each Tensor, the first
          dimension must be batch_size, otherwise, unknow errors may occur.
        beam_ids: A tensor with shape [batch_size, ] that used to gather states.

    Returns: A Tensor or a list/tuple of Tensors with the same structure
      as `states`.
    """

    def _gather(x):
        assert isinstance(x, tf.Tensor)
        return tf.gather(x, beam_ids)

    return nest.pack_sequence_as(
        states,
        nest.map_structure(
            _gather, nest.flatten(states)))


def finished_beam_one_entry_bias(on_entry, num_entries):
    """ Builds a bias vector to be added to log_probs of a finished beam.

    The returned vector with shape [`num_entries`, ]. Only the `on_entry`
    has value 0, and the others are FLOAT_MIN.

    For example, on_entry=3 and num_entries=6 get the vector
    [FLOAT_MIN, FLOAT_MIN, FLOAT_MIN, 0.0, FLOAT_MIN, FLOAT_MIN]

    Args:
        on_entry: A python integer.
        num_entries: A python integer.

    Returns: A bias vector.
    """
    pseudo_float_min = -1.0e9
    bias = tf.one_hot(
        [on_entry], num_entries,
        on_value=0., off_value=pseudo_float_min, dtype=tf.float32)
    return tf.squeeze(bias, axis=0)


def expand_to_beam_size(tensor, beam_size, axis=0):
    """ Stacks a given tensor `beam_size` times on a specific axis.

    For example, tensor=[1, 2, 3, 4], beam_size=3, axis=0 get the tensor
    [ [1, 2, 3, 4],
      [1, 2, 3, 4],
      [1, 2, 3, 4] ]

    tensor=[[1, 2, 3], [3, 4, 5]], beam_size=1, axis=1 get the tensor
    [ [[1, 2, 3]], [[3, 4, 5]] ]

    Args:
        tensor: A Tensor.
        beam_size: A python integer, the beam width.
        axis: A python integer.

    Returns: A Tensor.
    """
    tensor = tf.expand_dims(tensor, axis=axis)
    tile_dims = [1] * tensor.get_shape().ndims
    tile_dims[axis] = beam_size
    return tf.tile(tensor, tile_dims)


def compute_batch_indices(batch_size, k):
    """ Computes the i'th coordinate that contains the batch index for gathers.

    Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
    batch the beam item is in. This will create the i of the i,j coordinate
    needed for the gather.

    Args:
        batch_size: A python integer, the batch size.
        k: A python integer, the beam width.

    Returns: A Tensor.
    """
    # [beam_size, batch_size]: [[0, 1, 2,..., batch_size], [0, 1, 2,..., batch_size], ...]
    batch_pos = expand_to_beam_size(tf.range(batch_size), k)
    batch_pos = tf.transpose(batch_pos)
    return batch_pos


def compute_length_penalty(lengths, alpha):
    """ Computes length penalty, Referring
      to https://arxiv.org/abs/1609.08144.

    Args:
        lengths: The length tensor, with shape [n, ]
        alpha: The length penalty rate. Length penalty is given by
          (5+len(decode)/6) ^ -\alpha.
    Returns: The length penalty tensor.
    """
    if alpha is None or alpha < 0.0:
        alpha = 0.0
    return ((5.0 + tf.to_float(lengths)) / 6.0) ** (-alpha)


def process_beam_predictions(decoding_result, beam_size, alpha):
    """ Processes beam search results.

    Args:
        decoding_result: A dict returned by
        beam_size: An integer, the beam width.
        alpha: The length penalty rate.

    Returns: A dict.
    """
    # [_batch * _beam, ]
    log_probs = decoding_result["log_probs"][-1]
    length = decoding_result["decoding_length"]
    hypothesis = decoding_result["hypothesis"]
    if alpha is None or alpha < 0.0:
        penalty = 1.0 / tf.to_float(length)
    else:
        penalty = compute_length_penalty(length, alpha)
    scores = log_probs * penalty
    # [_batch * _beam, ] => [_batch, _beam]
    scores_flat = tf.reshape(scores, [-1, beam_size])
    # [_batch, _beam]
    top_scores, top_indices = tf.nn.top_k(scores_flat, k=beam_size)
    batch_beam_pos = compute_batch_indices(tf.shape(top_indices)[0], k=beam_size) * beam_size
    # [_batch * _beam, ]
    top_indices = tf.reshape(top_indices + batch_beam_pos, [-1])
    # [_batch * _beam, timesteps]
    sorted_hypothesis = tf.gather(hypothesis, top_indices)
    decoding_result["sorted_hypothesis"] = sorted_hypothesis
    decoding_result["scores"] = scores_flat
    decoding_result["sorted_argidx"] = top_indices
    return decoding_result
