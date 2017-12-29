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
""" Define learning rate decaying functions. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import inspect
import tensorflow as tf
from tensorflow.python.training import learning_rate_decay

from njunmt.utils.global_names import GlobalNames

# import all decay functions
DECAY_FNS = [
    x for x in learning_rate_decay.__dict__.values()
    if inspect.isfunction(x) and x.__name__.endswith("_decay")
    ]
for decay_fn in DECAY_FNS:
    setattr(sys.modules[__name__], decay_fn.__name__, decay_fn)


def loss_decay(learning_rate, global_step, decay_steps, decay_rate,
               staircase=False, name=None, **kwargs):
    """ The learning rate decaying function according to loss function.

    Defines variables manipulated by `LossMetricSpec`.

    Args:
        learning_rate: A tf Variable, the learning rate.
        global_step: A tf Variable, the global step. Not used.
        decay_steps: Not used.
        decay_rate: Not used.
        staircase: Not used.
        name: A string, the name scope.
        **kwargs: The function-specific parameters. For `loss_decay()`,
          it is "patience", a python integer to determine when to decay\
          the learning rate.

    Returns: A tuple `(decayed_lr, dict_of_vars)`, where `dict_of_vars`
      contains function-specific variables that will be added to tf.collections
      for other methods to refer to.
    """
    _ = global_step
    _ = decay_rate
    _ = decay_steps
    _ = staticmethod
    _ = staircase
    if "patience" not in kwargs or kwargs["patience"] is None:
        raise ValueError("\"patience\" should be provided when using loss_decay")
    with tf.name_scope(name, "LossDecay", [learning_rate]):
        div_factor = tf.get_variable(
            name=GlobalNames.LR_ANNEAL_DIV_FACTOR_NAME,
            shape=(), dtype=tf.float32,
            initializer=tf.constant_initializer(
                value=1., dtype=tf.float32),
            trainable=False)
        learning_rate = tf.convert_to_tensor(learning_rate, name=learning_rate)

        return (tf.div(learning_rate, div_factor),
                {GlobalNames.LR_ANNEAL_DIV_FACTOR_NAME: div_factor})


def noam_decay(learning_rate, global_step, decay_steps, decay_rate,
               staircase=False, name=None, **kwargs):
    """ Applies noam decay to learning rate as described in
    https://arxiv.org/abs/1706.03762.

    Args:
        learning_rate: A tf Variable, the learning rate.
        global_step: A tf Variable, the global step.
        decay_steps: A python integer, here means warmup steps.
        decay_rate: Not used.
        staircase: Not used.
        name: A string, the name scope.
        **kwargs: The function-specific parameters. For `noam_decay()`,
          it contains "dmodel" and "scale", where "dmodel" is the model
          size and "scale" is a float number to scale the result learning
          rate.

    Returns: The decayed learning rate.
    """
    _ = decay_rate
    _ = staircase
    if "dmodel" not in kwargs or kwargs["dmodel"] is None:
        raise ValueError("dmodel is required for noam_decay.")
    if "scale" not in kwargs or kwargs["scale"] is None:
        raise ValueError("scale is required for noam_decay.")

    with tf.name_scope(name, "noam_decay",
                       [learning_rate, global_step,
                        decay_steps]) as name:
        learning_rate = tf.convert_to_tensor(
            learning_rate, name="learning_rate", dtype=tf.float32)
        dtype = learning_rate.dtype
        dmodel = tf.convert_to_tensor(kwargs["dmodel"], dtype=dtype)
        scale = tf.convert_to_tensor(kwargs["scale"], dtype=dtype)
        global_step = tf.cast(global_step, dtype)
        decay_steps = tf.cast(decay_steps, dtype)

        return scale * dmodel ** -0.5 * tf.minimum(
            (global_step + 1.) * decay_steps ** -1.5, (global_step + 1.) ** -0.5)


def create_learning_rate_decay_fn(decay_type,
                                  decay_steps,
                                  decay_rate,
                                  start_decay_at=0,
                                  stop_decay_at=1e9,
                                  min_learning_rate=None,
                                  staircase=False,
                                  **kwargs):
    """ Creates a function that decays the learning rate.

    Args:
        decay_type: A decaying function name defined in `tf.train` or this file.
        decay_steps: How often to apply decay.
        decay_rate: The decay rate.
        start_decay_at: Don't decay before this step.
        stop_decay_at: Don't decay after this step.
        min_learning_rate: Don't decay below this number.
        staircase: Whether to apply decay in a discrete staircase,
          as opposed to continuous, fashion.
        **kwargs: Specific parameters for user-defined functions.

    Returns: A function that takes (learning_rate, global_step) as inputs
      and returns the learning rate for the given step.
      Or returns `None` if `decay_type` is empty or None.
    """
    if decay_type is None or decay_type == "":
        tf.logging.info("Optimizer: use fixed learning rate.")
        return None
    if not hasattr(sys.modules[__name__], decay_type):
        raise ValueError("Optimizer: unrecognized decay_type: {}.".format(decay_type))

    start_decay_at = tf.to_int32(start_decay_at)
    stop_decay_at = tf.to_int32(stop_decay_at)
    tf.logging.info("Optimizer: use decay_fn: {}".format(decay_type))

    def decay_fn(learning_rate, global_step):
        """The computed learning rate decay function.
        """
        global_step = tf.to_int32(global_step)
        decayed_learning_rate = eval(decay_type)(
            learning_rate=learning_rate,
            global_step=tf.minimum(global_step, stop_decay_at) - start_decay_at,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
            name="decayed_learning_rate",
            **kwargs)
        other_tensor_dict = {}
        if isinstance(decayed_learning_rate, tuple):
            decayed_learning_rate, other_tensor_dict = decayed_learning_rate

        final_lr = learning_rate_decay.piecewise_constant(
            x=global_step,
            boundaries=[start_decay_at],
            values=[learning_rate, decayed_learning_rate])

        if min_learning_rate:
            final_lr = tf.maximum(final_lr, min_learning_rate)
        return final_lr, other_tensor_dict

    return decay_fn


if __name__ == "__main__":
    with tf.Session() as sess:
        print(sess.run(noam_decay(
            0.2, 10, 16000,
            None, None, None,
            **{"dmodel": 512, "scale": 5000})))
