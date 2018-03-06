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
""" Define a wrapper class for optimizer and optimize function. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

from njunmt.utils.lr_decay import create_learning_rate_decay_fn
from njunmt.utils.misc import add_dict_to_collection
from njunmt.utils.configurable import Configurable
from njunmt.utils.constants import ModeKeys
from njunmt.utils.constants import Constants


def optimize(loss, opt_params, colocate_gradients_with_ops=False):
    """ Minimizes loss.

    Args:
        loss: The loss Tensor.
        opt_params: A dictionary of the parameters of the optimizer.
        colocate_gradients_with_ops: Argument passed to
          `tf.contrib.layers.optimize_loss`

    Returns: The train_op.
    """
    opt = OptimizerWrapper(opt_params)
    return opt.optimize(loss, colocate_gradients_with_ops)


def _get_optimizer(name, **params):
    """ Create optimizer.

    Args:
        name: A string, the name of the optimizer.
        **params: A dictionary of optimizer parameters.

    Returns: A Tensorflow optimizer.

    Raises:
        ValueError: if `name` is unknown.
    """
    if name in tf.contrib.layers.OPTIMIZER_CLS_NAMES.keys():
        return tf.contrib.layers.OPTIMIZER_CLS_NAMES[name](**params)
    if name == "LazyAdam":
        return tf.contrib.opt.LazyAdamOptimizer(**params)
    if name == "Adadelta":
        return tf.train.AdadeltaOptimizer(**params)
    raise ValueError("Unknown optimizer name: {}".format(name))


class OptimizerWrapper(Configurable):
    """ Define the wrapper class for creating optimizer. """
    def __init__(self, params):
        """ Initializes the parameters of the optimizer.

        Args:
            params: A dictionary of the parameters of the optimizer.
        """
        super(OptimizerWrapper, self).__init__(
            params=params, mode=ModeKeys.TRAIN,
            name=None, verbose=True)

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of the optimizer. """
        return {
            "optimizer.name": "Adam",
            "optimizer.learning_rate": 1e-4,
            "optimizer.params": {},  # Arbitrary parameters for the optimizer
            "optimizer.lr_decay": {
                "decay_type": None,
                "decay_steps": 100,
                "decay_rate": 0.99,
                "start_decay_at": 0,
                "stop_decay_at": sys.maxsize,
                "min_learning_rate": 1.0e-9,
                "staircase": False,
                "patience": None,  # for loss_decay
                "dmodel": None,  # for noam_decay
                "scale": 2.0  # for noam_decay
            },
            "optimizer.clip_gradients": 1.0,
            "optimizer.sync_replicas": 0,
            "optimizer.sync_replicas_to_aggregate": 0,
        }

    def optimize(self, loss, colocate_gradients_with_ops=False):
        """ Creates the optimizer with learning rate decaying, optimizes
        loss and return a train_op.

        Args:
            loss: The loss Tensor.
            colocate_gradients_with_ops: Argument passed to
              `tf.contrib.layers.optimize_loss`

        Returns: The train_op.
        """
        learning_rate = tf.get_variable(
            Constants.LEARNING_RATE_VAR_NAME,
            shape=(), dtype=tf.float32,
            initializer=tf.constant_initializer(
                value=self.params["optimizer.learning_rate"],
                dtype=tf.float32),
            trainable=False)
        name = self.params["optimizer.name"]
        tf.logging.info("use %s optimizer with initial learning rate=%f"
                        % (name, self.params["optimizer.learning_rate"]))

        global_step_tensor = tf.train.get_or_create_global_step()
        # create decay fn
        decay_fn = create_learning_rate_decay_fn(**self.params["optimizer.lr_decay"])
        other_tensor_dict = {}
        if decay_fn:  # apply learning rate decay
            learning_rate, other_tensor_dict = decay_fn(learning_rate, global_step_tensor)
        # add to collections
        other_tensor_dict[Constants.LEARNING_RATE_VAR_NAME] = learning_rate
        add_dict_to_collection(Constants.LEARNING_RATE_VAR_NAME, other_tensor_dict)
        tf.add_to_collection(Constants.DISPLAY_KEY_COLLECTION_NAME, "training_stats/learning_rate")
        tf.add_to_collection(Constants.DISPLAY_VALUE_COLLECTION_NAME, learning_rate)
        # create optimizer
        optimizer = _get_optimizer(name, learning_rate=learning_rate,
                                   **self.params["optimizer.params"])

        def _clip_gradients(grads_and_vars):
            """Clips gradients by global norm."""
            gradients, variables = zip(*grads_and_vars)
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, self.params["optimizer.clip_gradients"])
            return list(zip(clipped_gradients, variables))

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=global_step_tensor,
            learning_rate=None,  # self.params["optimizer.learning_rate"],
            learning_rate_decay_fn=None,
            clip_gradients=_clip_gradients if self.params["optimizer.clip_gradients"] > 0. else None,
            optimizer=optimizer,
            summaries=["learning_rate", "loss"],
            colocate_gradients_with_ops=colocate_gradients_with_ops)

        return train_op
