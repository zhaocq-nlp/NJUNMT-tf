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
""" Define transformer model as described in https://arxiv.org/abs/1706.03762."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import copy

from njunmt.utils.global_names import ModeKeys
from njunmt.utils.global_names import GlobalNames
from njunmt.models.sequence_to_sequence import SequenceToSequence


class Transformer(SequenceToSequence):
    """ Define transformer model as described in https://arxiv.org/abs/1706.03762.
    It is now reserved for old versions."""

    def __init__(self,
                 params,
                 mode,
                 vocab_source,
                 vocab_target,
                 name="tensor2tensor",
                 verbose=True):
        """ Initializes transformer parameters.

        Args:
            params: A dictionary of parameters to construct the
              model architecture.
            mode: A mode.
            vocab_source: An instance of `Vocab`.
            vocab_target: An instance of `Vocab`.
            name: The name of this decoder.
            verbose: Print model parameters if set True.
        """
        super(Transformer, self).__init__(params=params, mode=mode,
                                          vocab_source=vocab_source,
                                          vocab_target=vocab_target,
                                          name=name, verbose=verbose)
        assert self.params["encoder.class"] == "njunmt.encoders.transformer_encoder.TransformerEncoder", (
            "Transformer must use TransformerEncoder.")
        assert self.params["decoder.class"] == "njunmt.decoders.transformer_decoder.TransformerDecocder", (
            "Transformer must use TransformerDecoder.")

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this model. """
        t2t_default_params = copy.deepcopy(SequenceToSequence.default_params())
        t2t_default_params["initializer"] = "uniform_unit_scaling"
        t2t_default_params["inference.beam_size"] = 4
        t2t_default_params["encoder.class"] = "njunmt.encoders.transformer_encoder.TransformerEncoder"
        t2t_default_params["decoder.class"] = "njunmt.decoders.transformer_decoder.TransformerDecocder"
        if "bridge.class" in t2t_default_params:
            t2t_default_params.pop("bridge.class")
        if "bridge.params" in t2t_default_params:
            t2t_default_params.pop("bridge.params")
        return t2t_default_params

    def _compute_loss_t2t(self, logits, label_ids, label_length, target_modality):
        """ Computes loss via `target_modality`. The loss is further
        scaled by the ratio of the size of this batch to the size of
        the largest training batch ever.
        In Google's tensor2tensor code, it says:
        " The new data reader occasionally emits very small batches, which
          cause the examples in those batches to be grossly overweighted.
          We decrease the loss proportionally to the ratio of the size of this
          batch to the size of the largest training batch ever. "

        Rename this function to "_compute_loss" can turn on this setting.

        Args:
            logits: The logits Tensor with shape [timesteps, batch_size, target_vocab_size].
            label_ids: The labels Tensor with shape [batch_size, timesteps].
            label_length: The length of labels Tensor with shape [batch_size, ]
            target_modality: An instance of `Modality`.

        Returns: Loss on this batch of data, a tf.float32 scalar.
        """
        with tf.variable_scope(target_modality.name):
            loss = target_modality.loss(
                logits=logits, label_ids=label_ids, label_length=label_length)
        max_nonpadding_var = tf.get_variable(
            "max_nonpadding", shape=[],
            initializer=tf.ones_initializer(),
            trainable=False, dtype=tf.float32)
        if self.mode == ModeKeys.TRAIN:
            target_nonpadding_tokens = tf.to_float(tf.reduce_sum(label_length))
            max_nonpadding = tf.maximum(max_nonpadding_var, target_nonpadding_tokens)
            with tf.control_dependencies([tf.assign(max_nonpadding_var, max_nonpadding)]):
                small_batch_multilier = target_nonpadding_tokens / max_nonpadding
            # add to collection
            tf.add_to_collection(GlobalNames.DISPLAY_KEY_COLLECTION_NAME, "training_stats/small_batch_multilier")
            tf.add_to_collection(GlobalNames.DISPLAY_VALUE_COLLECTION_NAME, small_batch_multilier)
            tf.add_to_collection(GlobalNames.DISPLAY_KEY_COLLECTION_NAME, "training_stats/base_loss")
            tf.add_to_collection(GlobalNames.DISPLAY_VALUE_COLLECTION_NAME, loss)
            loss *= small_batch_multilier
        return loss

    def _create_bridge(self, encoder_output):
        """ Creates bridge between encoder and decoder.

        Overwrite the `_create_bridge()` in parent class because
        Transformer does not need bridge.

        Args:
            encoder_output: An instance of `collections.namedtuple`
              from `Encoder.encode()`.

        Returns: None.
        """
        return None