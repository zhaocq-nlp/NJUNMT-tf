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

import tensorflow as tf
from tensorflow.python.util import nest  # pylint: disable=E0611

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


def linear(args,
           output_size,
           bias,
           bias_initializer=None,
           kernel_initializer=None,
           kernel_name=_WEIGHTS_VARIABLE_NAME,
           bias_name=_BIAS_VARIABLE_NAME):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D/3D Tensor or a list of 2D/3D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      kernel_initializer: starting value to initialize the weight.

    Returns:
      A 2D/3D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):  # added by Chengqi
        ## capable for 3D tensor
        shape = args.get_shape()
        if shape.ndims > 2:
            scope = tf.get_variable_scope()
            with tf.variable_scope(scope) as outer_scope:
                weights = tf.get_variable(
                    kernel_name, [shape[-1].value, output_size],
                    dtype=args.dtype,
                    initializer=kernel_initializer)
                res = tf.tensordot(args, weights, [[shape.ndims - 1], [0]])

                if not bias:
                    return res
                with tf.variable_scope(outer_scope) as inner_scope:
                    inner_scope.set_partitioner(None)
                    if bias_initializer is None:
                        bias_initializer = tf.constant_initializer(0.0, dtype=args.dtype)
                    biases = tf.get_variable(
                        bias_name, [output_size],
                        dtype=args.dtype,
                        initializer=bias_initializer)
                return tf.nn.bias_add(res, biases)
        else:
            args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                             "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
        weights = tf.get_variable(
            kernel_name, [total_arg_size, output_size],
            dtype=dtype,
            initializer=kernel_initializer)

        if len(args) == 1:
            res = tf.matmul(args[0], weights)
        else:
            res = tf.matmul(tf.concat(args, 1), weights)
        if not bias:
            return res
        with tf.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            if bias_initializer is None:
                bias_initializer = tf.constant_initializer(0.0, dtype=dtype)
            biases = tf.get_variable(
                bias_name, [output_size],
                dtype=dtype,
                initializer=bias_initializer)
        return tf.nn.bias_add(res, biases)


def advanced_softmax(logits, mask=None):
    """
    compute softmax probability
    :param logits:
    :param mask:
    :return:
    """
    num_shapes = logits.get_shape().ndims
    if mask is not None:
        scores_exp = tf.exp(logits - tf.reduce_max(logits, axis=num_shapes - 1, keep_dims=True)) * mask
    else:
        scores_exp = tf.exp(logits - tf.reduce_max(logits, axis=num_shapes - 1, keep_dims=True))
    scores_sum = tf.reduce_sum(scores_exp, axis=num_shapes - 1, keep_dims=True)
    x_sm = scores_exp / scores_sum
    return x_sm


def advanced_log_softmax(logits):
    """
    compute softmax probability log results
    :param logits:
    :return:
    """
    return tf.log(advanced_softmax(logits))


def fflayer(inputs,
            output_size,
            activation=None,
            dropout_input_keep_prob=1.0,
            dropout_seed=None,
            bias=True,
            kernel_initializer=None,
            bias_initializer=None,
            name=None):
    """
    feed forward network for 2-dim matrix or 3-dim tensor
        capable for layer normalization and dropout
    :param inputs: input matrix or tensor
    :param output_size: output size
    :param activation: activation function, defualt None
    :param dropout_input_keep_prob:
    :param dropout_seed:
    :param bias: whether to use add a bias
    :param kernel_initializer:
    :param bias_initializer:
    :param name:
    :return:
    """
    scope = tf.get_variable_scope()
    with tf.variable_scope(name or scope):
        if (not isinstance(dropout_input_keep_prob, float)) or dropout_input_keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob=dropout_input_keep_prob, seed=dropout_seed)

        preact = linear(inputs, output_size=output_size, bias=bias,
                        kernel_name="W", bias_name="b",
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer)

    if activation is None:
        return preact
    return activation(preact)
