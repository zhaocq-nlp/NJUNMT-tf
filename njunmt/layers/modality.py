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
"""Modality class - defines the bottom and top of the model. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from njunmt.utils.configurable import Configurable
from njunmt.layers.common_layers import fflayer
from njunmt.layers.common_layers import add_sinusoids_timing_signal
from njunmt.training import loss_fns


class Modality(Configurable):
    """ Modality for sets of discrete symbols. """

    def __init__(self,
                 params,
                 mode,
                 vocab_size,
                 body_input_depth,
                 name=None,
                 verbose=True):
        """ Initializes modality parameters.

        Args:
            params: A dictionary of parameters to construct the
              decoder architecture.
            mode: A mode.
            vocab_size: The size of vocabulary.
            body_input_depth: The dimension of the embeddings.
            name: The name of this modality.
            verbose: Print decoder parameters if set True.
        """
        default_name = "symbol_modality_{}_{}".format(vocab_size, body_input_depth)
        super(Modality, self).__init__(
            params=params, mode=mode, verbose=verbose,
            name=name or default_name)
        self._vocab_size = vocab_size
        self._body_input_depth = body_input_depth

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this modality. """
        return {
            "multiply_embedding_mode": None,  # or sqrt_depth
            "share_embedding_and_softmax_weights": False,
            "dropout_logit_keep_prob": 1.0,
            "initializer": None,  # None for default, else random uniform, or random normal
            "loss": "crossentropy",
            "timing": None  # or sinusoids, or emb
        }

    def _check_parameters(self):
        assert self.params["timing"] in ["sinusoids", "emb", None], (
            "timing should be one of \"sinusoids\", \"emb\" or None")
        assert self.params["multiply_embedding_mode"] in ["sqrt_depth", None], (
            "multiply_embedding_mode should be one of \"sqrt_depth\" or None")
        assert self.params["loss"].lower() in ["crossentropy", "smoothing_crossentropy"], (
            "loss should be crossentropy or smoothing_crossentropy")
        assert self.params["initializer"] in ["random_uniform", "random_normal", None], (
            "initializer should be random_uniform, random_normal, or None")

    def _get_initializer(self, hidden_dim):
        """ Returns an initializer according the parameters.

        Args:
            hidden_dim: The dimension.

        Returns: An instance of `tf.Initializer`.
        """
        initializer = None
        if self.params["initializer"] == "random_uniform":
            scale = hidden_dim ** -0.5
            initializer = tf.random_uniform_initializer(-scale, scale)
        elif self.params["initializer"] == "random_normal":
            initializer = tf.random_normal_initializer(0.0, hidden_dim ** -0.5)
        return initializer

    def _get_weight(self, hidden_dim=None):
        """ Get a Tensor with shape [`self._vocab_size`, `hidden_dim`]

        Args:
            hidden_dim: The dimension. If not provided, use
              `self._body_input_depth` as default.

        Returns: A Tensor with shape [`self._vocab_size`, `hidden_dim`].
        """
        if hidden_dim is None:
            hidden_dim = self._body_input_depth
        initializer = self._get_initializer(hidden_dim)

        return tf.get_variable(
            name="weights",
            shape=(self._vocab_size, hidden_dim),
            initializer=initializer)

    def _get_position_weight(self, maximum_position=300, hidden_dim=None):
        """ Get a Tensor with shape [`maximum_postion`, `hidden_dim`]

        Args:
            maximum_position: The maximum number of positions.
            hidden_dim: The dimension. If not provided, use
              `self._body_input_depth` as default.

        Returns: A Tensor with shape [`maximum_position`, `hidden_dim`].
        """
        if hidden_dim is None:
            hidden_dim = self._body_input_depth
        initializer = self._get_initializer(hidden_dim)
        posi_emb_table = tf.get_variable(
            name="position_emb",
            shape=[maximum_position, hidden_dim],
            initializer=initializer)
        if self.params["multiply_embedding_mode"] == "sqrt_depth":
            posi_emb_table *= (hidden_dim ** 0.5)
        return posi_emb_table

    def _add_timing_signal(self, x, time):
        """ Adds timing signal (also known as position encoding) to `x`

        Args:
            x: A Tensor with shape [batch_size, dim] or [batch_size, timesteps, dim]
            time: An integer indicating the position. If `x` has shape
              [batch_size, timesteps, dim], set None.

        Returns: A Tensor of the same shape of `x`.

        Raises:
            ValueError: if `time`==None and `x` is a 2-d Tensor.
        """
        timing = self.params["timing"]
        if timing is None:
            return x
        x_ndims = x.get_shape().ndims
        if x_ndims == 2 and time is None:
            raise ValueError("\"time\" should be provided when input x has 2-dims")
        if timing == "sinusoids":
            return add_sinusoids_timing_signal(x=x, time=time)
        if timing == "emb":
            position_emb_table = self._get_position_weight()
            if x_ndims == 2:
                position = tf.convert_to_tensor(time, dtype=tf.int32)
            elif x_ndims == 3:
                position = tf.range(tf.shape(x)[1])
            else:
                raise ValueError("need a Tensor with rank 2 or 3")
            position_emb = tf.nn.embedding_lookup(position_emb_table, position)
            return x + tf.expand_dims(position_emb, axis=0)

    @property
    def top_dimension(self):
        """ Returns the size of vocabulary. """
        return self._vocab_size

    def top(self, top_features):
        """ Computes logits on the top layer.

        Args:
            top_features: A Tensor.

        Returns: A logits Tensor.
        """
        feature_last_dim = top_features.get_shape().as_list()[-1]
        if self.params["share_embedding_and_softmax_weights"]:
            assert feature_last_dim == self._body_input_depth, \
                "when shared_embedding_and_softmax_weights, dim_logits should be equal to input_depth"
            scope_name = "shared"
            with tf.variable_scope(scope_name, reuse=True):
                var = tf.transpose(self._get_weight(feature_last_dim), [1, 0])
        else:
            scope_name = "softmax"
            var = None
        logits = fflayer(top_features, output_size=self.top_dimension, handle=var,
                         activation=None, name=scope_name,
                         dropout_input_keep_prob=self.params["dropout_logit_keep_prob"])
        return logits

    def bottom_simple(self, x, name, reuse, time=None):
        """ Embeds the symbols.

        Args:
            x: A 1/2-d Tensor to be embedded.
            name: A string.
            reuse: The parameter to `tf.variable_scope()`.
            time: An integer indicating the position. If `x` has shape
              [batch_size, timesteps], set None.

        Returns: A 2/3-d Tensor according to `x`.
        """
        with tf.variable_scope(name, reuse=reuse):
            var = self._get_weight()
            ret = tf.nn.embedding_lookup(var, x)
            if self.params["multiply_embedding_mode"] == "sqrt_depth":
                ret *= self._body_input_depth ** 0.5
            return self._add_timing_signal(ret, time)

    def bottom(self, x, time=None):
        """ Embeds the symbols (on the source side).

        Args:
            x: A 1/2-d Tensor to be embedded.
            time: An integer indicating the position. If `x` has shape
              [batch_size, timesteps], set None.

        Returns: A 2/3-d Tensor according to `x`.
        """
        if self.params["share_embedding_and_softmax_weights"]:
            return self.bottom_simple(x, "shared", reuse=None, time=time)
        return self.bottom_simple(x, "input_emb", reuse=None, time=time)

    def targets_bottom(self, x, time=None):
        """ Embeds the symbols (on the target side).

        Args:
            x: A 1/2-d Tensor to be embedded.
            time: An integer indicating the position. If `x` has shape
              [batch_size, timesteps], set None.

        Returns: A 2/3-d Tensor according to `x`.
        """
        name = "target_emb"
        if self.params["share_embedding_and_softmax_weights"]:
            name = "shared"
        try:
            return self.bottom_simple(x, name, reuse=True, time=time)
        except ValueError:
            # perhaps there were no inputs, and this is a new variable.
            return self.bottom_simple(x, name, reuse=None, time=time)

    def loss(self, logits, label_ids, label_length):
        """ Computes loss.

        Args:
            logits: A logits Tensor with shape [timesteps, batch_size, vocab_size].
            label_ids: The gold symbol ids, a Tensor with shape [batch_size, timesteps].
            label_length: The true symbols lengths, a Tensor with shape [batch_size, ].

        Returns: A tf.float32 scalar.
        """
        loss = getattr(loss_fns, self.params["loss"])(
            logits=logits,
            # transposed targets: [timesteps, batch_size]
            targets=tf.transpose(label_ids, [1, 0]),
            sequence_length=label_length)
        return loss
