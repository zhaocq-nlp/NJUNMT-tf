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
""" Implement transformer decoder as described in https://arxiv.org/abs/1706.03762. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.util import nest
from collections import namedtuple

from njunmt.utils.constants import ModeKeys
from njunmt.decoders.decoder import dynamic_decode
from njunmt.decoders.decoder import initialize_cache
from njunmt.decoders.decoder import Decoder
from njunmt.layers.common_layers import dropout_wrapper
from njunmt.layers.common_layers import layer_preprocess
from njunmt.layers.common_layers import layer_postprocessing
from njunmt.layers.common_layers import transformer_ffn_layer
from njunmt.layers.common_attention import MultiHeadAttention
from njunmt.layers.common_attention import attention_bias_lower_triangle


class TransformerDecoder(Decoder):
    """ Implement transformer decoder as described
    in https://arxiv.org/abs/1706.03762. """

    def __init__(self,
                 params,
                 mode,
                 name=None,
                 verbose=True):
        """ Initializes decoder parameters.

        Args:
            params: A dictionary of parameters to construct the
              decoder architecture.
            mode: A mode.
            name: The name of this decoder.
            verbose: Print decoder parameters if set True.
        """
        super(TransformerDecoder, self).__init__(params, mode, name, verbose)

        self._self_attention_layers = []
        self._encdec_attention_layers = []
        for layer in range(self.params["num_layers"]):
            self._self_attention_layers.append(
                MultiHeadAttention(self.params["selfattention.params"], self.mode))
            self._encdec_attention_layers.append(
                MultiHeadAttention(self.params["attention.params"], self.mode))
        if self.mode == ModeKeys.TRAIN:
            self._DecoderOutputSpec = namedtuple(
                "TransformerOutput", "decoder_hidden")
        elif self.mode == ModeKeys.EVAL:
            self._DecoderOutputSpec = namedtuple(
                "TransformerOutput", "decoder_hidden decoder_self_attention encoder_decoder_attention")
        else:
            self._DecoderOutputSpec = namedtuple(
                "TransformerOutput", "decoder_hidden encoder_decoder_attention")

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of TransformerDecoder. """
        return {
            "num_layers": 6,
            "attention.params": {},  # Arbitrary parameters for the enc-dec attention layer
            "selfattention.params": {},  # Arbitrary parameters for the self-attention layer
            "num_filter_units": 2048,
            "num_hidden_units": 512,
            "dropout_relu_keep_prob": 0.9,
            "layer_preprocess_sequence": "n",
            "layer_postprocess_sequence": "da",
            "layer_prepostprocess_dropout_keep_prob": 0.9
        }

    @property
    def output_dtype(self):
        """ Returns a `collections.namedtuple`,
        the definition of decoder output types. """
        if self.mode == ModeKeys.TRAIN:
            return self._DecoderOutputSpec(
                decoder_hidden=tf.float32)
        elif self.mode == ModeKeys.EVAL:
            return self._DecoderOutputSpec(
                decoder_hidden=tf.float32,
                decoder_self_attention=[tf.float32] * self.params["num_layers"],
                encoder_decoder_attention=[tf.float32] * self.params["num_layers"])
        else:
            return self._DecoderOutputSpec(
                decoder_hidden=tf.float32,
                encoder_decoder_attention=[tf.float32] * self.params["num_layers"])

    def merge_top_features(self, decoder_output):
        """ Merges features of decoder top layers, as the input
        of softmax layer.

        Here is the same as the hidden state of the last layer
        of the transformer decoder.

        Args:
            decoder_output: An instance of `collections.namedtuple`
              whose element types are defined by `output_dtype`
              property.

        Returns: A instance of `tf.Tensor`, as the input of
          softmax layer.
        """
        return decoder_output.decoder_hidden

    def decode(self, encoder_output, bridge, helper,
               target_to_embedding_fn,
               outputs_to_logits_fn,
               **kwargs):
        """ Decodes one sample.

        Args:
            encoder_output: An instance of `collections.namedtuple`
              from `Encoder.encode()`.
            bridge: None.
            helper: An instance of `Feedback` that samples next
              symbols from logits.
            target_to_embedding_fn: A callable, converts target ids to
              embeddings.
            outputs_to_logits_fn: A callable, converts decoder outputs
              to logits.

        Returns: A tuple `(decoder_output, decoder_status)`. The
          `decoder_output` is an instance of `collections.namedtuple`
          whose element types are defined by `output_dtype` property.
          For mode=INFER, the `decoder_status` is a dict containing
          hypothesis, log probabilities, beam ids and decoding length.
          For mode=TRAIN/EVAL, the `decoder_status` is a `tf.Tensor`
          indicating logits (computed by `target_modality`), of shape
          [timesteps, batch_size, vocab_size].
        """
        if bridge is not None and self.verbose:
            tf.logging.info(
                "TransformerDecoder ignores bridge: {}".format(bridge.name))

        if self.mode == ModeKeys.TRAIN or self.mode == ModeKeys.EVAL:
            assert hasattr(helper, "label_ids"), (
                "helper ({}) for TransformerDecoder when mode=TRAIN/EVAL "
                "should provide attr \"label_ids\"".format(type(helper)))
            # prepare decoder input
            label_ids = getattr(helper, "label_ids")  # [batch_size, max_len_trg]
            batch_size = tf.shape(label_ids)[0]
            target_sos_ids = tf.tile([helper.vocab.sos_id], [batch_size])
            target_sos_ids = tf.reshape(target_sos_ids, [batch_size, 1])
            label_ids = tf.concat([target_sos_ids, label_ids], axis=1)[:, :-1]
            decoder_inputs = target_to_embedding_fn(label_ids)
            with tf.variable_scope(self.name):
                cache = self.prepare(encoder_output, None, helper)
                outputs, decoder_self_attention, encdec_attention \
                    = self._transform(decoder_inputs, cache)  # [batch_size, time, dim]
                if self.mode == ModeKeys.TRAIN:
                    final_outputs = self._DecoderOutputSpec(
                        decoder_hidden=outputs)
                else:
                    final_outputs = self._DecoderOutputSpec(
                        decoder_hidden=outputs,
                        # transpose to [length_q, batch_size, num_heads length_k]
                        decoder_self_attention=nest.map_structure(
                            lambda x: tf.transpose(x, [2, 0, 1, 3]), decoder_self_attention),
                        encoder_decoder_attention=nest.map_structure(
                            lambda x: tf.transpose(x, [2, 0, 1, 3]), encdec_attention))
                decoder_top_features = self.merge_top_features(final_outputs)
            # do transpose to fit loss function, [time, batch_size, dim]
            decoder_top_features = tf.transpose(decoder_top_features, [1, 0, 2])
            logits = outputs_to_logits_fn(decoder_top_features)  # [time, batch_size, vocab_size]
            return final_outputs, logits
        outputs, infer_status = dynamic_decode(
            decoder=self, encoder_output=encoder_output,
            bridge=None, helper=helper,
            target_to_embedding_fn=target_to_embedding_fn,
            outputs_to_logits_fn=outputs_to_logits_fn,
            **kwargs)
        return outputs, infer_status

    def prepare(self, encoder_output, bridge, helper):
        """ Prepares for `step()` function.
        Do
            1. acquire attention information from `encoder_output`;

        Args:
            encoder_output: An instance of `collections.namedtuple`
              from `Encoder.encode()`.
            bridge: None.
            helper: An instance of `Feedback` that samples next
              symbols from logits.

        Returns: A dict containing decoder RNN states, pre-projected attention
          keys, attention values and attention length, and will be passed
          to `step()` function.
        """
        _ = bridge
        attention_values = encoder_output.attention_values
        attention_length = encoder_output.attention_length
        if hasattr(encoder_output, "attention_bias"):
            attention_bias = encoder_output.attention_bias
        else:
            attention_bias = MultiHeadAttention.attention_length_to_bias(None, attention_length)

        # initialize cache
        if self.mode == ModeKeys.INFER:
            decoding_states = {}
            batch_size = tf.shape(attention_values)[0]
            depth = self._self_attention_layers[0].attention_value_depth
            if depth < 0:
                # TODO please check when code goes into this condition
                depth = tf.shape(attention_values)[2]
            # initialize decoder self attention keys/values
            for l in range(self.params["num_layers"]):
                keys = tf.zeros([batch_size, 0, depth])
                values = tf.zeros([batch_size, 0, depth])
                # Ensure shape invariance for tf.while_loop.
                keys.set_shape([None, None, depth])
                values.set_shape([None, None, depth])
                with tf.variable_scope("layer_%d" % l):
                    with tf.variable_scope("encdec_attention"):
                        with tf.variable_scope(self._encdec_attention_layers[l].name):
                            preproj_keys, preproj_values = self._encdec_attention_layers[l] \
                                .compute_kv(attention_values)
                decoding_states["layer_{}".format(l)] = {
                    "self_attention": {"keys": keys, "values": values},
                    "encdec_attention": {"attention_keys": preproj_keys,
                                         "attention_values": preproj_values}}
        else:
            decoding_states = None

        init_cache = initialize_cache(
            decoding_states=decoding_states,
            memory=attention_values,
            memory_bias=attention_bias)
        return init_cache

    def step(self, decoder_input, cache):
        """ Decodes one step.

        Args:
            decoder_input: The decoder input for this timestep.
              A Tensor, with shape [batch_size, dmodel].
            cache: A dict containing decoding states at previous
              timestep, attention values and attention length.

        Returns: A tuple `(cur_decoder_outputs, cur_cache)` at this timestep.
          The `cur_decoder_outputs` must be an instance of `collections.namedtuple`
          whose element types are defined by `output_dtype` property. The
          `cur_cache` must have the same structure with `cache`.

        """
        # decoder self attention: [batch_size, num_heads, length_q, length_k]
        outputs, decoder_self_attention, encdec_attention = \
            self._transform(tf.expand_dims(decoder_input, axis=1), cache)
        final_outputs = self._DecoderOutputSpec(
            decoder_hidden=outputs[:, -1, :],
            # decoder_self_attention=[tf.squeeze(att, axis=2) for att in decoder_self_attention],
            encoder_decoder_attention=[tf.squeeze(att, axis=2) for att in encdec_attention])
        # loop on decoder_state, actually it is not used
        return final_outputs, cache

    def _transform(self, decoder_inputs, cache):
        """ Decodes one step

        Args:
            decoder_inputs: The decoder input for this timestep,
              A Tensor, with shape [batch_size, timesteps, dmodel].
              Note that when mode==INFER, timesteps=1.
            cache: A dict containing decoding states at previous
              timestep, attention values and attention length.

        Returns: A transformed Tensor.
        """
        # [batch_size, max_len_src, dim]
        encdec_attention_values = cache["memory"]
        # [batch_size, 1, 1, max_len_src]
        encdec_attention_bias = cache["memory_bias"]

        decoder_self_attention_scores = []
        encdec_attention_scores = []

        # decoder_self_attention_bias: [1, 1, max_len_trg, max_len_trg]
        decoder_self_attention_bias = attention_bias_lower_triangle(
            tf.shape(decoder_inputs)[1])
        x = dropout_wrapper(decoder_inputs, self.params["layer_prepostprocess_dropout_keep_prob"])
        for layer in range(self.params["num_layers"]):
            layer_name = "layer_{}".format(layer)
            layer_cache = None if cache["decoding_states"] is None \
                else cache["decoding_states"][layer_name]
            selfatt_cache = None if layer_cache is None \
                else layer_cache["self_attention"]
            encdecatt_cache = None if layer_cache is None \
                else layer_cache["encdec_attention"]
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    # self attention layer
                    w_y, y = self._self_attention_layers[layer].build(
                        query=None,
                        memory=layer_preprocess(
                            x=x, process_sequence=self.params["layer_preprocess_sequence"],
                            dropout_keep_prob=self.params["layer_prepostprocess_dropout_keep_prob"]),
                        memory_bias=decoder_self_attention_bias,
                        cache=selfatt_cache)
                    # [batch_size, num_heads, length_q, length_k]
                    decoder_self_attention_scores.append(w_y)
                    # apply dropout, layer norm, residual
                    x = layer_postprocessing(
                        x=y, previous_x=x,
                        process_sequence=self.params["layer_postprocess_sequence"],
                        dropout_keep_prob=self.params["layer_prepostprocess_dropout_keep_prob"])
                with tf.variable_scope("encdec_attention"):
                    # encoder-decoder attention
                    w_y, y = self._encdec_attention_layers[layer].build(
                        query=layer_preprocess(
                            x=x, process_sequence=self.params["layer_preprocess_sequence"],
                            dropout_keep_prob=self.params["layer_prepostprocess_dropout_keep_prob"]),
                        memory=encdec_attention_values,
                        memory_bias=encdec_attention_bias,
                        cache=encdecatt_cache)
                    # [batch_size, num_heads, length_q, length_k]
                    encdec_attention_scores.append(w_y)
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
                        pad_remover=None,
                        dropout_relu_keep_prob=self.params["dropout_relu_keep_prob"])
                    # apply dropout, layer norm, residual
                    x = layer_postprocessing(
                        x=y, previous_x=x,
                        process_sequence=self.params["layer_postprocess_sequence"],
                        dropout_keep_prob=self.params["layer_prepostprocess_dropout_keep_prob"])
        x = layer_preprocess(
            x=x, process_sequence=self.params["layer_preprocess_sequence"],
            dropout_keep_prob=self.params["layer_prepostprocess_dropout_keep_prob"])
        return x, decoder_self_attention_scores, encdec_attention_scores
