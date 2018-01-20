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
from collections import namedtuple

from njunmt.utils.global_names import ModeKeys
from njunmt.decoders.decoder import dynamic_decode
from njunmt.decoders.decoder import Decoder
from njunmt.layers.common_layers import dropout_wrapper
from njunmt.layers.common_layers import layer_preprocess
from njunmt.layers.common_layers import layer_postprocessing
from njunmt.layers.common_layers import transformer_ffn_layer
from njunmt.layers.common_attention import MultiHeadAttention
from njunmt.layers.common_attention import attention_bias_ignore_padding
from njunmt.layers.common_attention import embedding_to_padding
from njunmt.layers.common_attention import attention_bias_lower_triangle
from njunmt.layers.common_attention import multihead_attention_layer


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
        self._DecoderOutputSpec = namedtuple(
            "TransformerOutput", "decoder_hidden")

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
        return self._DecoderOutputSpec(
            decoder_hidden=tf.float32)

    def inputs_prepost_processing_fn(self):
        """ This function is for generalization purpose. For `tf.while_loop`
        in `dynamic_decode` function, reshape the input tensor to
        [batch_size, timesteps, dmodel], before it is passed to
        `step()` fn, and set the input tensor to [None, dmodel]
        before it is passed to `tf.while_loop`.

        Returns: A tuple `(preprocessing_fn, postprocessing_fn)`.
        """
        preprocessing_fn = lambda time, inputs: \
            tf.reshape(inputs, [-1, time + 1, inputs.get_shape().as_list()[-1]])

        def postprocessing_fn(prev_inputs, predicted_inputs):
            next_inputs = predicted_inputs
            if prev_inputs is not None:
                next_inputs = tf.concat(
                    [prev_inputs, tf.expand_dims(predicted_inputs, axis=1)],
                    axis=1)
            dmodel = next_inputs.get_shape().as_list()[-1]
            next_inputs = tf.reshape(next_inputs, [-1, dmodel])
            next_inputs.set_shape([None, dmodel])
            return next_inputs

        return preprocessing_fn, postprocessing_fn

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
               target_modality):
        """ Decodes one sample.

        Args:
            encoder_output: An instance of `collections.namedtuple`
              from `Encoder.encode()`.
            bridge: None.
            helper: An instance of `Feedback` that samples next
              symbols from logits.
            target_modality: An instance of `Modality`, that deals
              with transformations from symbols to tensors or from
              tensors to symbols (the decoder top and bottom layer).

        Returns: A tuple `(decoder_output, decoder_status)`. The
          `decoder_output` is an instance of `collections.namedtuple`
          whose element types are defined by `output_dtype` property.
          For mode=INFER, the `decoder_status` is an instance of
          `collections.namedtuple` whose element types are defined by
          `BeamSearchStateSpec`, indicating the status of beam search.
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
            with tf.variable_scope(target_modality.name):
                decoder_inputs = target_modality.targets_bottom(label_ids)
            with tf.variable_scope(self.name):
                _, decoding_params = self.prepare(encoder_output, None, helper)
                outputs = self._transform(decoder_inputs, decoding_params)  # [batch_size, time, dim]
                final_outputs = self._DecoderOutputSpec(
                    decoder_hidden=outputs)
                decoder_top_features = self.merge_top_features(final_outputs)
            # do transpose to fit loss function, [time, batch_size, dim]
            decoder_top_features = tf.transpose(decoder_top_features, [1, 0, 2])
            with tf.variable_scope(target_modality.name):
                logits = target_modality.top(decoder_top_features)  # [time, batch_size, vocab_size]
            return final_outputs, logits
        outputs, infer_status = dynamic_decode(decoder=self, encoder_output=encoder_output, bridge=None, helper=helper,
                                               target_modality=target_modality)
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
        Returns: A tuple `(init_decoder_states, decoding_params)`.
          `init_decoder_states` is a scalar float32 value, `tf.while_loop`
          will loop on this value but do nothing (for generalization
          purpose). `decoding_params` is a tuple containing attention
          values and will be passed to `step()` function.
        """
        _ = bridge
        attention_values = encoder_output.attention_values
        attention_length = encoder_output.attention_length
        if hasattr(encoder_output, "attention_bias"):
            attention_bias = encoder_output.attention_bias
        else:
            attention_bias = MultiHeadAttention.attention_length_to_bias(None, attention_length)
        decoding_params = (attention_values, attention_length, attention_bias)
        # use a constant as the placeholder for while_loop
        return tf.constant(1.0, dtype=tf.float32), decoding_params

    def step(self, decoder_input, decoder_states, decoding_params):
        """ Decodes one step.

        Args:
            decoder_input: The decoder input for this timestep, an
              instance of `tf.Tensor`, [batch_size, timesteps, dmodel].
            decoder_states: Anything which is ignored here.
            decoding_params: The same as `decoding_params` returned
              from `prepare()` function.

        Returns: A tuple `(cur_decoder_outputs, cur_decoder_states)`
          at this timestep. The `cur_decoder_outputs` must be an
          instance of `collections.namedtuple` whose element types
          are defined by `output_dtype` property. The
          `cur_decoder_states` is the same as the `decoder_states`.

        """
        outputs = self._transform(decoder_input, decoding_params)
        final_outputs = self._DecoderOutputSpec(decoder_hidden=outputs[:, -1, :])
        # loop on decoder_state, actually it is not used
        return final_outputs, decoder_states

    def _transform(self, decoder_inputs, decoding_params):
        """ Decodes one step

        Args:
            decoder_input: The decoder input for this timestep, an
              instance of `tf.Tensor`, [batch_size, timesteps, dmodel].
            decoding_params: The same as `decoding_params` returned
              from `prepare()` function.

        Returns: A Tensor, the transformed hidden
          state of TransformerDecoder.
        """
        # [batch_size, max_len_src, dim]
        encdec_attention_values = decoding_params[0]
        # [batch_size, ]
        # encdec_attention_length = decoding_params[1]
        # [batch_size, 1, 1, max_len_src]
        encdec_attention_bias = decoding_params[2]

        # decoder_self_attention_bias: [1, 1, max_len_trg, max_len_trg]
        decoder_self_attention_bias = attention_bias_lower_triangle(
            tf.shape(decoder_inputs)[1])
        x = dropout_wrapper(decoder_inputs, self.params["layer_prepostprocess_dropout_keep_prob"])
        for layer in range(self.params["num_layers"]):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    # self attention layer
                    w_y, y = multihead_attention_layer(
                        params=self.params["selfattention.params"],
                        mode=self.mode,
                        query_antecedent=None,
                        memory_antecedent=layer_preprocess(
                            x=x, process_sequence=self.params["layer_preprocess_sequence"],
                            dropout_keep_prob=self.params["layer_prepostprocess_dropout_keep_prob"]),
                        memory_bias=decoder_self_attention_bias)
                    # apply dropout, layer norm, residual
                    x = layer_postprocessing(
                        x=y, previous_x=x,
                        process_sequence=self.params["layer_postprocess_sequence"],
                        dropout_keep_prob=self.params["layer_prepostprocess_dropout_keep_prob"])
                with tf.variable_scope("encdec_attention"):
                    # encoder-decoder attention
                    w_y, y = multihead_attention_layer(
                        params=self.params["attention.params"],
                        mode=self.mode,
                        query_antecedent=layer_preprocess(
                            x=x, process_sequence=self.params["layer_preprocess_sequence"],
                            dropout_keep_prob=self.params["layer_prepostprocess_dropout_keep_prob"]),
                        memory_antecedent=encdec_attention_values,
                        memory_bias=encdec_attention_bias)
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
        return layer_preprocess(
            x=x, process_sequence=self.params["layer_preprocess_sequence"],
            dropout_keep_prob=self.params["layer_prepostprocess_dropout_keep_prob"])
