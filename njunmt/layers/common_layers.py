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
""" Define common layers, e.g. conv, fflayer and so on. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from njunmt.utils.algebra_ops import linear


def dropout_wrapper(x, keep_prob, seed=None):
    """ A wrapper function for `tf.nn.dropout`

    Args:
        x: A Tensor.
        keep_prob: A float, the probability that each
          element is kept.
        seed: A Python integer. Used to create random seeds.

    Returns: A `tf.Tensor` of the same shape of `x`.
    """
    if keep_prob < 1.0:
        return tf.nn.dropout(x, keep_prob=keep_prob, seed=seed)
    return x


def layer_preprocess(x, process_sequence, dropout_keep_prob):
    """ Applies layer preprocessing.

    See layer_prepostprocess() for details.

    Args:
        x: The layer input, an instance of `tf.Tensor`.
        process_sequence: The process sequence for the layer input.
        dropout_keep_prob: A float, the probability that each
          element is kept.

    Returns: A Tensor of the same shape of `x`.
    """
    assert "a" not in process_sequence, (
        "No residual connections allowed in layer_preprocess_sequence")
    return layer_prepostprocess(
        x=x,
        previous_x=None,
        process_sequence=process_sequence,
        dropout_keep_prob=dropout_keep_prob,
        name="layer_prepostprocess")


def layer_prepostprocess(x, previous_x, process_sequence, dropout_keep_prob, name=None):
    """ Applies a sequence of functions to the input or output
    of a layer.

    The sequence is specified as a string which may contain the
    following characters:
      a: add previous_x
      n: apply normalization
      d: apply dropout

    For example, if sequence=="dna", then the output is
       previous_x + normalize(dropout(x))

    Args:
        x: A Tensor to be transformed.
        previous_x: A Tensor, to be added as a
          residual connection ('a').
        process_sequence: The process sequence, string.
        dropout_keep_prob: A float, the probability that each
          element is kept.
        name: A string.

    Returns: A Tensor of the same shape of `x`.
    """
    with tf.variable_scope(name or "layer_prepostprocess"):
        if process_sequence is None or process_sequence.lower() == "none":
            return x
        for c in process_sequence:
            if c == "a":
                x += previous_x
            elif c == "n":
                x = norm_layer(x)
            elif c == "d":
                x = dropout_wrapper(x, keep_prob=dropout_keep_prob)
            else:
                raise ValueError("Unknown sequence step {}".format(c))
        return x


def layer_postprocessing(x, previous_x, process_sequence, dropout_keep_prob):
    """ Applies layer postprocessing.

    See layer_prepostprocess() for details.

    Args:
        x: The layer output, an instance of `tf.Tensor`.
        previous_x: The layer input, an instance of `tf.Tensor`,
          for residual purpose.
        process_sequence: The process sequence for the layer input.
        dropout_keep_prob: A float, the probability that each
          element is kept.

    Returns: A Tensor of the same shape of `x`.
    """
    return layer_prepostprocess(
        x=x,
        previous_x=previous_x,
        process_sequence=process_sequence,
        dropout_keep_prob=dropout_keep_prob)


def fflayer(inputs,
            output_size,
            handle=None,
            activation=None,
            dropout_input_keep_prob=1.0,
            dropout_seed=None,
            bias=True,
            kernel_initializer=None,
            bias_initializer=None,
            name=None):
    """ Applies feed forward transform for a 2-d matrix or
     3-d tensor.

    Args:
        inputs: A Tensor of 2-d or 3-d, [..., dim]
        output_size: An integer.
        handle: A Tensor. If provided, use it as the weight matrix.
        activation: The activation function.
        dropout_input_keep_prob: A float, the probability that each
          element in `inputs` is kept.
        dropout_seed: A Python integer. Used to create random seeds.
        bias: Whether to add a bias vector.
        kernel_initializer: The initializer of kernel weight.
        bias_initializer: The initializer of bias vector.
        name: A string.

    Returns: A Tensor of shape [..., `output_size`]
    """
    scope = tf.get_variable_scope()
    with tf.variable_scope(name or scope):
        inputs = dropout_wrapper(
            inputs, keep_prob=dropout_input_keep_prob,
            seed=dropout_seed)

        preact = linear(inputs, output_size=output_size,
                        handle=handle,
                        bias=bias,
                        kernel_name="W", bias_name="b",
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer)

    if activation is None:
        return preact
    return activation(preact)


def conv2d(inputs, filters, kernel_size, **kwargs):
    """ Convolution layer.

    Args:
        inputs: A 4-d tensor.
        filters: The filter size.
        kernel_size: A tuple of kernel sizes.
        **kwargs:

    Returns: A tensor.
    """
    assert inputs.get_shape().ndims == 4
    if "name" in kwargs:
        name = kwargs.pop("name")
    else:
        name = "conv"
    ret = tf.layers.conv2d(inputs, filters, kernel_size,
                           name=name, **kwargs)
    return ret


def conv1d(inputs, filters, kernel_size, dilation_rate=1, **kwargs):
    """ Convolution layer.

    Args:
        inputs: A 3-d tensor.
        filters: The filter size.
        kernel_size: The kernel size.
        dilation_rate:
        **kwargs:
    Returns: A tensor.
    """
    assert inputs.get_shape().ndims == 3
    return tf.squeeze(
        conv2d(
            tf.expand_dims(inputs, 2),
            filters, (kernel_size, 1),
            dilation_rate=(dilation_rate, 1),
            **kwargs),
        axis=2)


def transformer_ffn_layer(x,
                          filter_size,
                          output_size,
                          pad_remover=None,
                          kernel_size=(1, 1),
                          second_kernel_size=(1, 1),
                          dropout_relu_keep_prob=1.0,
                          **kwargs):
    """ Applies the position-wise feed-forward as described
    in https://arxiv.org/abs/1706.03762

    Args:
        x: A Tensor, to be transformed, of shape
          [batch_size, max_sequence_length, hidden_size].
        filter_size: The hidden size of relu layer.
        output_size: The hidden size of the second linear layer, always the
          same as the hidden size of `x`.
        pad_remover: An expert_utils.PadRemover object tracking the padding
          positions. If provided, the padding is removed before applying
          the convolution, and restored afterward. This can give a significant
          speedup (says Google's tensor2tensor code).
        kernel_size: The kernel size of the first convolution.
        second_kernel_size: The kernel size of the second convolution.
        dropout_relu_keep_prob: A float, the probability that each
          element in the relu layer is kept.
        **kwargs:

    Returns: A Tensor of shape
      [batch_size, max_sequence_length, output_size]
    """
    if pad_remover:
        original_shape = tf.shape(x)
        x = tf.reshape(x, tf.concat([[-1], x.get_shape().as_list()[2:]], axis=0))
        x = tf.expand_dims(pad_remover.remove(x), axis=0)
    conv_output = conv_hidden_relu(
        x,
        filter_size,
        output_size,
        kernel_size=kernel_size,
        second_kernel_size=second_kernel_size,
        dropout_relu_keep_prob=dropout_relu_keep_prob,
        **kwargs)
    if pad_remover:
        conv_output = tf.reshape(
            pad_remover.restore(tf.squeeze(conv_output, axis=0)),
            original_shape)
    return conv_output


def conv_hidden_relu(inputs,
                     hidden_size,
                     output_size,
                     kernel_size=(1, 1),
                     second_kernel_size=(1, 1),
                     dropout_relu_keep_prob=1.0,
                     **kwargs):
    """ Hidden layer with RELU activation followed by linear projection.

    Args:
        inputs: A Tensor, to be transformed.
        hidden_size: The hidden size of the relu layer.
        output_size: The output size.
        kernel_size: The kernel size of the relu layer.
        second_kernel_size: The kernel size of the linear layer.
        dropout_relu_keep_prob: A float, the probability that each
          element in the relu layer is kept.
        **kwargs:

    Returns: A Tensor after transformation.
    """
    name = kwargs.pop("name") if "name" in kwargs else None
    with tf.variable_scope(name or "conv_hidden_relu"):
        if inputs.get_shape().ndims == 3:
            is_3d = True
            inputs = tf.expand_dims(inputs, 2)
        else:
            is_3d = False
        h = conv2d(inputs, hidden_size, kernel_size,
                   activation=tf.nn.relu, name="conv1")
        h = dropout_wrapper(h, keep_prob=dropout_relu_keep_prob)
        ret = conv2d(h, output_size, second_kernel_size,
                     name="conv2", **kwargs)
        if is_3d:
            ret = tf.squeeze(ret, 2)
    return ret


def add_sinusoids_timing_signal(x, time, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.

    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).

    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    This function is originally copied from Google's Tensor2Tensor code
    and modified to hold the capability for add timing signal at the
    specific time.

    Args:
      x: a Tensor with shape [batch, length, channels]
      min_timescale: a float
      max_timescale: a float

    Returns: A Tensor the same shape as x.
    """
    channels = x.get_shape().as_list()[-1]
    if x.get_shape().ndims == 3:  # [batch_size, timesteps, dim]
        length = tf.shape(x)[1]
        position = tf.to_float(tf.range(length))
    elif x.get_shape().ndims == 2:  # [batch_size, dim]
        length = 1
        position = tf.to_float(tf.range(time, time + 1))
    else:
        raise ValueError("need a Tensor with rank 2 or 3")
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    if x.get_shape().ndims == 3:
        signal = tf.reshape(signal, [1, length, channels])
    else:
        signal = tf.reshape(signal, [1, channels])
    return x + signal


def norm_layer(x, gain=1.0, shift=0.0, name="ln", reuse=None, epsilon=1.e-6):
    """ Applies layer normalization on the last dimension of the
    input tensor.

    Args:
        x: A Tensor.
        gain: A float. The initial scale of layer normalization.
        shift: A float. The initial bias of layer normalization.
        name: A string.
        reuse: The parameter passed to `tf.variable_scope()`
        epsilon: A float to avoid zeros.

    Returns: A Tensor of the same shape of `x`.
    """
    filters = x.get_shape()[-1]
    with tf.variable_scope(name, default_name="layer_norm",
                           values=[x], reuse=reuse):
        # Initialize beta and gamma for use by layer_norm.
        gamma = tf.get_variable(
            "gamma", shape=[filters], initializer=tf.constant_initializer(gain))
        beta = tf.get_variable(
            "beta", shape=[filters], initializer=tf.constant_initializer(shift))
        mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        normalized = norm_x * gamma + beta
        return normalized
