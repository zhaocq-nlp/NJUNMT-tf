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
""" Define loss functions. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from njunmt.utils.misc import label_smoothing
from njunmt.utils.misc import deprecated
from njunmt.utils.expert_utils import PadRemover
from njunmt.layers.common_attention import embedding_to_padding


@deprecated
def crossentropy_avgall(logits, targets, sequence_length):
    """ Computes cross entropy loss of a batch of data.

    The final loss is first averaged by the length of each
    sequence and then averaged by the batch size.

    Args:
        logits: The logits Tensor with shape [timesteps, batch_size, vocab_size].
        targets: The gold labels Tensor with shape [timesteps, batch_size].
        sequence_length: The length of `targets`, [batch_size, ]

    Returns: Loss sum and weight sum.
    """
    # [timesteps, batch_size]
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=targets)

    # [timesteps, batch_size]
    loss_mask = tf.transpose(
        tf.sequence_mask(
            lengths=tf.to_int32(sequence_length),
            maxlen=tf.to_int32(tf.shape(targets)[0]),
            dtype=tf.float32), [1, 0])

    losses = losses * loss_mask
    # average loss
    avg_length = tf.to_float(sequence_length)
    loss_by_time = tf.reduce_sum(losses, axis=0) / avg_length
    loss_sum = tf.reduce_sum(loss_by_time)
    return loss_sum, tf.to_float(tf.shape(sequence_length)[0])


@deprecated
def smoothing_crossentropy_avgall(logits, targets, sequence_length):
    """ Computes cross entropy loss of a batch of data with label smoothing.

    The final loss is averaged by the length of each
    sequence and then averaged by the batch size.

    Args:
        logits: The logits Tensor with shape [timesteps, batch_size, vocab_size].
        targets: The gold labels Tensor with shape [timesteps, batch_size].
        sequence_length: The length of `targets`, [batch_size, ]

    Returns: Loss sum and weight sum.
    """
    soft_targets, normalizing = label_smoothing(targets, logits.get_shape().as_list()[-1])
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=soft_targets) - normalizing
    # [timesteps, batch_size]
    loss_mask = tf.transpose(
        tf.sequence_mask(
            lengths=tf.to_int32(sequence_length),
            maxlen=tf.to_int32(tf.shape(targets)[0]),
            dtype=tf.float32), [1, 0])
    losses = losses * loss_mask
    # average loss
    avg_length = tf.to_float(sequence_length)
    loss_by_time = tf.reduce_sum(losses, axis=0) / avg_length
    loss_sum = tf.reduce_sum(loss_by_time)
    return loss_sum, tf.to_float(tf.shape(sequence_length)[0])


def crossentropy_t(logits, targets, sequence_length):
    """ Computes cross entropy loss of a batch of data.

    The final loss is averaged by the number of tokens in the batch.

    Args:
        logits: The logits Tensor with shape [timesteps, batch_size, vocab_size].
        targets: The gold labels Tensor with shape [timesteps, batch_size].
        sequence_length: The length of `targets`, [batch_size, ]

    Returns: Loss sum and weight sum.
    """
    # [timesteps, batch_size]
    padding = tf.transpose(embedding_to_padding(tf.shape(logits)[0], sequence_length), [1, 0])
    padremover = PadRemover(padding)
    # [-1, vocab_size]
    logits = tf.reshape(logits, tf.concat([[-1], logits.get_shape().as_list()[2:]], axis=0))
    logits = padremover.remove(logits)
    targets = padremover.remove(tf.reshape(targets, [-1]))
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=targets)
    loss_sum = tf.reduce_sum(losses)
    return loss_sum, tf.to_float(tf.shape(losses)[0])


def crossentropy(logits, targets, sequence_length):
    """ Computes cross entropy loss of a batch of data. (Not averaged by batch_size)

    The final loss is averaged by the number of samples in the batch.

    Args:
        logits: The logits Tensor with shape [timesteps, batch_size, vocab_size].
        targets: The gold labels Tensor with shape [timesteps, batch_size].
        sequence_length: The length of `targets`, [batch_size, ]

    Returns: Loss sum and weight sum.
    """
    # [timesteps, batch_size]
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=targets)

    # [timesteps, batch_size]
    loss_mask = tf.transpose(
        tf.sequence_mask(
            lengths=tf.to_int32(sequence_length),
            maxlen=tf.to_int32(tf.shape(targets)[0]),
            dtype=tf.float32), [1, 0])

    losses = losses * loss_mask
    loss_sum = tf.reduce_sum(losses)
    return loss_sum, tf.to_float(tf.shape(sequence_length)[0])


def smoothing_crossentropy(logits, targets, sequence_length):
    """ Computes cross entropy loss of a batch of data with label smoothing.

    The final loss is averaged by the number of tokens in the batch.

    Args:
        logits: The logits Tensor with shape [timesteps, batch_size, vocab_size].
        targets: The gold labels Tensor with shape [timesteps, batch_size].
        sequence_length: The length of `targets`, [batch_size, ]

    Returns: Loss sum and weight sum.
    """
    soft_targets, normalizing = label_smoothing(targets, logits.get_shape().as_list()[-1])
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=soft_targets) - normalizing
    # [timesteps, batch_size]
    loss_mask = tf.transpose(
        tf.sequence_mask(
            lengths=tf.to_int32(sequence_length),
            maxlen=tf.to_int32(tf.shape(targets)[0]),
            dtype=tf.float32), [1, 0])
    losses = losses * loss_mask
    loss_sum = tf.reduce_sum(losses)
    return loss_sum, tf.to_float(tf.shape(sequence_length)[0])


def smoothing_crossentropy_t(logits, targets, sequence_length):
    """ Computes cross entropy loss of a batch of data with label smoothing.

    The final loss is averaged by the number of tokens in the batch.

    Args:
        logits: The logits Tensor with shape [timesteps, batch_size, vocab_size].
        targets: The gold labels Tensor with shape [timesteps, batch_size].
        sequence_length: The length of `targets`, [batch_size, ]

    Returns: Loss sum and weight sum.
    """
    soft_targets, normalizing = label_smoothing(targets, logits.get_shape().as_list()[-1])
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=soft_targets) - normalizing
    # [timesteps, batch_size]
    loss_mask = tf.transpose(
        tf.sequence_mask(
            lengths=tf.to_int32(sequence_length),
            maxlen=tf.to_int32(tf.shape(targets)[0]),
            dtype=tf.float32), [1, 0])
    losses = losses * loss_mask
    loss_sum = tf.reduce_sum(losses)
    return loss_sum, tf.to_float(tf.reduce_sum(sequence_length))
