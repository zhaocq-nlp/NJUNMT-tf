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
""" Define some advanced algebra operations. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

from njunmt.utils.misc import deprecated

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


@deprecated
def _layer_norm(inputs,
                center=True,
                scale=True,
                activation_fn=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                begin_norm_axis=-1,
                begin_params_axis=-1,
                scope=None):
    """ copy from tensorflow version 1.3 (diff from version1.2.1),
        with default begin_norm_axis set to -1, that is, we only normalize the last dim

    Adds a Layer Normalization layer.
    Based on the paper:
      "Layer Normalization"
      Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
      https://arxiv.org/abs/1607.06450.
    Can be used as a normalizer function for conv2d and fully_connected.
    Given a tensor `inputs` of rank `R`, moments are calculated and normalization
    is performed over axes `begin_norm_axis ... R - 1`.  Scaling and centering,
    if requested, is performed over axes `begin_shift_axis .. R - 1`.
    By default, `begin_norm_axis = 1` and `begin_params_axis = -1`,
    meaning that normalization is performed over all but the first axis
    (the `HWC` if `inputs` is `NHWC`), while the `beta` and `gamma` trainable
    parameters are calculated for the rightmost axis (the `C` if `inputs` is
    `NHWC`).  Scaling and recentering is performed via broadcast of the
    `beta` and `gamma` parameters with the normalized tensor.
    The shapes of `beta` and `gamma` are `inputs.shape[begin_params_axis:]`,
    and this part of the inputs' shape must be fully defined.
    Args:
      inputs: A tensor having rank `R`. The normalization is performed over
        axes `begin_norm_axis ... R - 1` and centering and scaling parameters
        are calculated over `begin_params_axis ... R - 1`.
      center: If True, add offset of `beta` to normalized tensor. If False, `beta`
        is ignored.
      scale: If True, multiply by `gamma`. If False, `gamma` is
        not used. When the next layer is linear (also e.g. `nn.relu`), this can be
        disabled since the scaling can be done by the next layer.
      activation_fn: Activation function, default set to None to skip it and
        maintain a linear activation.
      reuse: Whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: Optional collections for the variables.
      outputs_collections: Collections to add the outputs.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      begin_norm_axis: The first normalization dimension: normalization will be
        performed along dimensions `begin_norm_axis : rank(inputs)`
      begin_params_axis: The first parameter (beta, gamma) dimension: scale
        and centering parameters will have dimensions
        `begin_params_axis : rank(inputs)` and will be broadcast with the
        normalized inputs accordingly.
      scope: Optional scope for `variable_scope`.
    Returns:
      A `Tensor` representing the output of the operation, having the same
      shape and dtype as `inputs`.
    Raises:
      ValueError: If the rank of `inputs` is not known at graph build time,
        or if `inputs.shape[begin_params_axis:]` is not fully defined at
        graph build time.
    """
    with vs.variable_scope(scope, 'LayerNorm', [inputs],
                           reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        inputs_shape = inputs.shape
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        dtype = inputs.dtype.base_dtype
        if begin_norm_axis < 0:
            begin_norm_axis = inputs_rank + begin_norm_axis
        if begin_params_axis >= inputs_rank or begin_norm_axis >= inputs_rank:
            raise ValueError(
                'begin_params_axis (%d) and begin_norm_axis (%d) '
                'must be < rank(inputs) (%d)'
                % (begin_params_axis, begin_norm_axis, inputs_rank))
        params_shape = inputs_shape[begin_params_axis:]
        if not params_shape.is_fully_defined():
            raise ValueError(
                'Inputs %s: shape(inputs)[%s:] is not fully defined: %s' % (
                    inputs.name, begin_params_axis, inputs_shape))
        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            beta_collections = utils.get_variable_collections(variables_collections,
                                                              'beta')
            beta = variables.model_variable(
                'beta',
                shape=params_shape,
                dtype=dtype,
                initializer=init_ops.zeros_initializer(),
                collections=beta_collections,
                trainable=trainable)
        if scale:
            gamma_collections = utils.get_variable_collections(variables_collections,
                                                               'gamma')
            gamma = variables.model_variable(
                'gamma',
                shape=params_shape,
                dtype=dtype,
                initializer=init_ops.ones_initializer(),
                collections=gamma_collections,
                trainable=trainable)
        # Calculate the moments on the last axis (layer activations).
        norm_axes = list(range(begin_norm_axis, inputs_rank))
        mean, variance = nn.moments(inputs, norm_axes, keep_dims=True)
        # Compute layer normalization using the batch_normalization function.
        variance_epsilon = 1e-9
        outputs = nn.batch_normalization(
            inputs, mean, variance, offset=beta, scale=gamma,
            variance_epsilon=variance_epsilon)
        outputs.set_shape(inputs_shape)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope,
                                           outputs)


def linear(args,
           output_size,
           bias,
           handle=None,
           bias_initializer=None,
           kernel_initializer=None,
           kernel_name=_WEIGHTS_VARIABLE_NAME,
           bias_name=_BIAS_VARIABLE_NAME):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    This function is originally copied from rnn_cell_impl.py and add
    the capability to deal with 3-D matrix.
    Args:
        args: a 2D/3D Tensor or a list of 2D/3D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        handle: A Tensor. If provided, use it
          as the weight matrix.
        bias_initializer: starting value to initialize the bias
          (default is all zeros).
        kernel_initializer: starting value to initialize the weight.

    Returns: A 2D/3D Tensor with shape [batch x output_size] equal to
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
            scope = vs.get_variable_scope()
            with vs.variable_scope(scope) as outer_scope:
                if handle is None:
                    weights = vs.get_variable(
                        kernel_name, [shape[-1].value, output_size],
                        dtype=args.dtype,
                        initializer=kernel_initializer)
                else:
                    assert output_size == handle.get_shape().as_list()[-1], \
                        "ouput_size should be the same as the last dimension of handle tensor"
                    weights = handle

                res = math_ops.tensordot(args, weights, [[shape.ndims - 1], [0]])

                if not bias:
                    return res
                with vs.variable_scope(outer_scope) as inner_scope:
                    inner_scope.set_partitioner(None)
                    if bias_initializer is None:
                        bias_initializer = init_ops.constant_initializer(0.0, dtype=args.dtype)
                    biases = vs.get_variable(
                        bias_name, [output_size],
                        dtype=args.dtype,
                        initializer=bias_initializer)
                return nn_ops.bias_add(res, biases)
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
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        if handle is None:
            weights = vs.get_variable(
                kernel_name, [total_arg_size, output_size],
                dtype=dtype,
                initializer=kernel_initializer)
        else:
            assert output_size == handle.get_shape().as_list()[-1], \
                "ouput_size should be the same as the last dimension of handle tensor"
            weights = handle
        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            if bias_initializer is None:
                bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
            biases = vs.get_variable(
                bias_name, [output_size],
                dtype=dtype,
                initializer=bias_initializer)
        return nn_ops.bias_add(res, biases)


def advanced_softmax(logits, mask=None):
    """ Computes softmax function manually.

    Avoids numeric overflow.

    Args:
        logits: A Tensor. The softmax will apply on the last dimension of it.
        mask: A Tensor with the same shape as `logits`.

    Returns: The softmax results.
    """
    num_shapes = logits.get_shape().ndims
    if mask is not None:
        scores_exp = math_ops.exp(logits - math_ops.reduce_max(logits, axis=num_shapes - 1, keep_dims=True)) * mask
    else:
        scores_exp = math_ops.exp(logits - math_ops.reduce_max(logits, axis=num_shapes - 1, keep_dims=True))
    scores_sum = math_ops.reduce_sum(scores_exp, axis=num_shapes - 1, keep_dims=True)
    x_sm = scores_exp / scores_sum
    return x_sm


def advanced_log_softmax(logits):
    """ Computes log of softmax probabilities.

    Args:
        logits: A Tensor. The softmax will apply on the last dimension of it.

    Returns: The log softmax results.
    """
    return math_ops.log(advanced_softmax(logits))


def advanced_reduce_sum(values, values_length, axis):
    """ Reudces sum at `axis`.

    Args:
        values: A tensor with shape [batch, time, dim] or [time, batch, dim]
        values_length: A tensor with shape [batch,]
        axis: The axis indicating time, 0/1.

    Returns: The reduced tensor with shape [batch, dim]
    """
    # [batch_size, time]
    mask = array_ops.sequence_mask(
        lengths=values_length,
        maxlen=array_ops.shape(values)[axis],
        dtype=dtypes.float32)
    if axis == 0:
        mask = array_ops.transpose(mask, perm=[1, 0])
    masked_values = values * array_ops.expand_dims(mask, axis=2)
    return math_ops.reduce_sum(masked_values, axis=axis)


def advanced_reduce_mean(values, values_length, axis):
    reduced_sum = advanced_reduce_sum(values, values_length, axis)
    return reduced_sum / array_ops.expand_dims(math_ops.to_float(values_length), axis=1)


def split_last_dimension(x, n):
    """ Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.

    Args:
        x: A Tensor with shape [..., m].
        n: A python integer.

    Returns: A Tensor with shape [..., n, m/n].

    """
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = array_ops.reshape(x, array_ops.concat([array_ops.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return ret


def combine_last_two_dimensions(x):
    """ Reshape x so that the last two dimension become one.

    The inverse of `split_last_dimension()`.

    Args:
        x: A Tensor with shape [..., a, b].

    Returns: A Tensor with shape [..., ab].
    """
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    ret = array_ops.reshape(x, array_ops.concat([array_ops.shape(x)[:-2], [-1]], 0))
    ret.set_shape(new_shape)
    return ret
