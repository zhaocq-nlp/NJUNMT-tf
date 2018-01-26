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
""" Implement transformer encoder as described in https://arxiv.org/abs/1706.03762. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import namedtuple

from njunmt.encoders.encoder import Encoder
from njunmt.utils.global_names import ModeKeys
from njunmt.utils.expert_utils import PadRemover
from njunmt.layers.common_layers import dropout_wrapper
from njunmt.layers.common_layers import layer_postprocessing
from njunmt.layers.common_layers import layer_preprocess
from njunmt.layers.common_layers import transformer_ffn_layer
from njunmt.layers.common_attention import MultiHeadAttention
from njunmt.layers.common_attention import attention_bias_to_padding


class TransformerEncoder(Encoder):
    """ Define transformer encoder as described
    in https://arxiv.org/abs/1706.03762. """

    def __init__(self, params, mode, name=None, verbose=True):
        """ Initializes the parameters of the encoder.

        Args:
            params: A dictionary of parameters to construct the
              encoder architecture.
            mode: A mode.
            name: The name of this encoder.
            verbose: Print encoder parameters if set True.
        """
        super(TransformerEncoder, self).__init__(params=params, mode=mode, name=name, verbose=verbose)

        self._self_attention_layers = []
        for layer in range(self.params["num_layers"]):
            self._self_attention_layers.append(
                MultiHeadAttention(self.params["selfattention.params"], self.mode))

        if self.mode == ModeKeys.TRAIN:
            self.encoder_output_tuple_type = namedtuple(
                "EncoderOutput",
                "outputs attention_values attention_length")
        else:
            self.encoder_output_tuple_type = namedtuple(
                "EncoderOutput",
                "outputs attention_values attention_length encoder_self_attention")


    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this encoder. """
        return {
            "num_layers": 6,
            "selfattention.params": {},  # Arbitrary parameters for the self-attention layer
            "num_filter_units": 2048,
            "num_hidden_units": 512,
            "dropout_relu_keep_prob": 0.9,
            "layer_preprocess_sequence": "n",
            "layer_postprocess_sequence": "da",
            "layer_prepostprocess_dropout_keep_prob": 0.9
        }

    def encode(self, feature_ids, feature_length, input_modality, **kwargs):
        """ Encodes the inputs.

        Args:
            feature_ids: A Tensor, [batch_size, max_features_length].
            feature_length: A Tensor, [batch_size, ].
            input_modality: An instance of `Modality`.
            **kwargs:

        Returns: An instance of `collections.namedtuple`.
        """
        with tf.variable_scope(input_modality.name):
            inputs = input_modality.bottom(feature_ids)
        with tf.variable_scope(self.name) as vs:
            # [batch_size, 1, 1, timesteps], FLOAT_MIN for padding, 0.0 for non-padding
            encoder_attention_bias = MultiHeadAttention.attention_length_to_bias(inputs, feature_length)
            outputs, enc_self_attention = self._transform(inputs, encoder_attention_bias, scope=vs, **kwargs)
            if self.mode == ModeKeys.TRAIN:
                encoder_output = self.encoder_output_tuple_type(
                    # [batch_size, timesteps, dim]
                    outputs=outputs,
                    attention_values=outputs,
                    attention_length=feature_length)
            else:
                encoder_output = self.encoder_output_tuple_type(
                    # [batch_size, timesteps, dim]
                    outputs=outputs,
                    attention_values=outputs,
                    attention_length=feature_length,
                    # a list of Tensors, [batch_size, num_heads, length_q, length_k]
                    encoder_self_attention=enc_self_attention)
            return encoder_output

    def _transform(self, inputs, encoder_self_attention_bias, **kwargs):
        """ Encodes the inputs.

        Args:
            inputs: A Tensor, [batch_size, timesteps, d_model]
            encoder_self_attention_bias: A Tensor, FLOAT_MIN
              for padding, 0 for non-padding, [batch_size, 1, 1, timesteps].
            **kwargs:

        Returns: A Tensor, the transformed hidden
          state of TransformerEncoder, [batch_size, timesteps, d_model].

        """
        input_padding = attention_bias_to_padding(encoder_self_attention_bias)
        pad_remover = PadRemover(input_padding)
        x = dropout_wrapper(inputs, self.params["layer_prepostprocess_dropout_keep_prob"])
        encoder_self_attention_scores = []
        for layer in range(self.params["num_layers"]):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    # self attention layer
                    w_y, y = self._self_attention_layers[layer].build(
                        query=None,
                        memory=layer_preprocess(
                            x=x, process_sequence=self.params["layer_preprocess_sequence"],
                            dropout_keep_prob=self.params["layer_prepostprocess_dropout_keep_prob"]),
                        memory_bias=encoder_self_attention_bias)
                    encoder_self_attention_scores.append(w_y)
                    # apply dropout, layer norm, residual
                    x = layer_postprocessing(
                        x=y, previous_x=x,
                        process_sequence=self.params["layer_postprocess_sequence"],
                        dropout_keep_prob=self.params["layer_prepostprocess_dropout_keep_prob"])
                with tf.variable_scope("ffn"):
                    y = transformer_ffn_layer(
                        x=layer_preprocess(
                            x=x, process_sequence=self.params["layer_preprocess_sequence"],
                            dropout_keep_prob=self.params["layer_prepostprocess_dropout_keep_prob"]),
                        filter_size=self.params["num_filter_units"],
                        output_size=self.params["num_hidden_units"],
                        pad_remover=pad_remover,
                        dropout_relu_keep_prob=self.params["dropout_relu_keep_prob"])
                    # apply dropout, layer norm, residual
                    x = layer_postprocessing(
                        x=y, previous_x=x,
                        process_sequence=self.params["layer_postprocess_sequence"],
                        dropout_keep_prob=self.params["layer_prepostprocess_dropout_keep_prob"])
        x = layer_preprocess(
            x=x, process_sequence=self.params["layer_preprocess_sequence"],
            dropout_keep_prob=self.params["layer_prepostprocess_dropout_keep_prob"])
        return x, encoder_self_attention_scores
