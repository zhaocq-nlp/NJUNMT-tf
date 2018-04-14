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
""" Define base sequence-to-sequence model. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys
import six
from abc import ABCMeta
import tensorflow as tf

import njunmt
from njunmt.layers.modality import Modality
from njunmt.utils import bridges
from njunmt.utils import feedback
from njunmt.utils.configurable import Configurable
from njunmt.utils.configurable import deep_merge_dict
from njunmt.utils.constants import Constants
from njunmt.utils.constants import ModeKeys
from njunmt.utils.beam_search import process_beam_predictions
from njunmt.utils.misc import set_fflayers_layer_norm

# import all bridges
BRIDGE_CLSS = [
    x for x in bridges.__dict__.values()
    if inspect.isclass(x) and issubclass(x, bridges.Bridge)
    ]
for bri in BRIDGE_CLSS:
    setattr(sys.modules[__name__], bri.__name__, bri)


@six.add_metaclass(ABCMeta)
class SequenceToSequence(Configurable):
    """ Base Sequence-to-Sequence Class """

    __MODEL_COUNTER = 0

    def __init__(self,
                 params,
                 mode,
                 vocab_source,
                 vocab_target,
                 name="sequence_to_sequence",
                 verbose=True):
        """ Initializes model parameters.

        Args:
            params: A dictionary of parameters to construct the
              model architecture.
            mode: A mode.
            vocab_source: An instance of `Vocab`.
            vocab_target: An instance of `Vocab`.
            name: The name of this decoder.
            verbose: Print model parameters if set True.
        """
        super(SequenceToSequence, self).__init__(
            params=params, mode=mode, verbose=False, name=name)
        self._vocab_source = vocab_source
        self._vocab_target = vocab_target
        self._verbose = verbose
        set_fflayers_layer_norm(self.params["fflayers.layer_norm"])
        # create Network components
        self._input_modality, self._target_modality = self._create_modalities()
        self._encoder = self._create_encoder()
        self._decoder = self._create_decoder()
        self._encoder_decoder_bridge = self._create_bridge()

    @staticmethod
    def create_input_fields(mode):
        """ Creates tf placeholders and add input data status to tf
          collections for displaying.

        Returns: A dictionary of placeholders.
        """
        SequenceToSequence.__MODEL_COUNTER += 1
        inp = dict()
        feature_ids = tf.placeholder(
            tf.int32, shape=(None, None),
            name="{}_{}".format(Constants.FEATURE_IDS_NAME, SequenceToSequence.__MODEL_COUNTER - 1))
        feature_length = tf.placeholder(
            tf.int32, shape=(None,),
            name="{}_{}".format(Constants.FEATURE_LENGTH_NAME, SequenceToSequence.__MODEL_COUNTER - 1))
        inp[Constants.FEATURE_IDS_NAME] = feature_ids
        inp[Constants.FEATURE_LENGTH_NAME] = feature_length
        if mode == ModeKeys.INFER:
            return inp

        label_ids = tf.placeholder(
            tf.int32, shape=(None, None),
            name="{}_{}".format(Constants.LABEL_IDS_NAME, SequenceToSequence.__MODEL_COUNTER - 1))
        label_length = tf.placeholder(
            tf.int32, shape=(None,),
            name="{}_{}".format(Constants.LABEL_LENGTH_NAME, SequenceToSequence.__MODEL_COUNTER - 1))
        inp[Constants.LABEL_IDS_NAME] = label_ids
        inp[Constants.LABEL_LENGTH_NAME] = label_length
        return inp

    def _create_modalities(self):
        """ Creates source and target modalities.

        Returns: A tuple `(input_modality, target_modality)`.
        """
        input_modality_params = deep_merge_dict(
            self.params["modality.params"], self.params["modality.source.params"])
        input_modality = Modality(
            params=input_modality_params,
            mode=self.mode,
            vocab_size=self._vocab_source.vocab_size,
            body_input_depth=self.params["embedding.dim.source"],
            name="input_symbol_modality",
            verbose=self.verbose)
        target_modality_params = deep_merge_dict(
            self.params["modality.params"], self.params["modality.target.params"])
        target_modality = Modality(
            params=target_modality_params,
            mode=self.mode,
            vocab_size=self._vocab_target.vocab_size,
            body_input_depth=self.params["embedding.dim.target"],
            name="target_symbol_modality",
            verbose=self.verbose)
        return input_modality, target_modality

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this model. """
        return {
            "fflayers.layer_norm": False,
            "encoder.class": "njunmt.encoders.rnn_encoder.StackBidirectionalRNNEncoder",
            "encoder.params": {},  # Arbitrary parameters for the encoder
            "bridge.class": "ZeroBridge",
            "bridge.params": {},  # Arbitrary parameters for the bridge
            "decoder.class": "njunmt.decoders.rnn_decoder.CondAttentionDecoder",
            "decoder.params": {},  # Arbitrary parameters for the decoder
            "embedding.dim.source": 512,
            "embedding.dim.target": 512,
            "modality.source.params": {},  # Arbitrary parameters for the modality
            "modality.target.params": {},  # Arbitrary parameters for the modality
            "modality.params": {},  # Arbitrary parameters for the modality
            "source.reverse": False,
            "inference.beam_size": 10,
            "inference.maximum_labels_length": 150,
            "inference.length_penalty": -1.0,
            "initializer": "random_uniform"}

    def get_variable_initializer(self):
        """ Returns the default initializer of the model scope.

        Returns: A tf initializer.
        """
        dmodel = self.params["embedding.dim.target"]
        if self.params["initializer"] == "random_uniform":
            return tf.random_uniform_initializer(
                -dmodel ** -0.5, dmodel ** -0.5)
        elif self.params["initializer"] == "normal_unit_scaling":
            return tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_AVG", uniform=False)
        elif self.params["initializer"] == "uniform_unit_scaling":
            return tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_AVG", uniform=True)
        else:
            raise ValueError("Unrecognized initializer: {}".format(self.params["initializer"]))

    def build(self, input_fields):
        """ Builds the sequence-to-sequence model.

        This function calls many inner functions to build each component
        of the model.

        Args:
            input_fields: A dictionary of placeholders.

        Returns: Model output. See _pack_output() for more details.
        """
        with tf.variable_scope(self._name, initializer=self.get_variable_initializer()):
            encoder_output = self._encode(input_fields=input_fields)
            decoder_output, decoding_res = self._decode(
                encoder_output=encoder_output,
                input_fields=input_fields)

            final_outputs = self._pack_output(
                encoder_output, decoder_output, decoding_res, **input_fields)
        return final_outputs

    def _compute_loss(self, logits, label_ids, label_length):
        """ Computes loss via `target_modality`.

        Args:
            logits: The logits Tensor with shape [timesteps, batch_size, target_vocab_size].
            label_ids: The labels Tensor with shape [batch_size, timesteps].
            label_length: The length of labels Tensor with shape [batch_size, ]

        Returns: Loss sum and weight sum.
        """
        with tf.variable_scope(self._target_modality.name):
            return self._target_modality.loss(
                logits=logits, label_ids=label_ids, label_length=label_length)

    def _decode(self, encoder_output, input_fields):
        """ Builds helper and calls decoder's `decode` method.

        Args:
            encoder_output: An instance of `collections.namedtuple`
              from `Encoder.encode()`.
            input_fields: A dictionary of placeholders.

        Returns: The results of decoding. For more details, see
          `Decoder.decode()`.
        """
        if self.mode == ModeKeys.TRAIN \
                or self.mode == ModeKeys.EVAL:
            label_ids = input_fields[Constants.LABEL_IDS_NAME]
            label_length = input_fields[Constants.LABEL_LENGTH_NAME]
            helper = feedback.TrainingFeedback(
                vocab=self._vocab_target, label_ids=label_ids, label_length=label_length)

        else:  # self.mode == tf.contrib.learn.ModeKeys.INFER
            helper = feedback.BeamFeedback(
                vocab=self._vocab_target,
                batch_size=tf.shape(input_fields[Constants.FEATURE_IDS_NAME])[0],
                maximum_labels_length=self.params["inference.maximum_labels_length"],
                beam_size=self.params["inference.beam_size"],
                alpha=self.params["inference.length_penalty"])
        decoder_output, decoding_res = self._decoder.decode(
            encoder_output, self._encoder_decoder_bridge, helper,
            self._target_to_embedding_fn,
            self._outputs_to_logits_fn,
            beam_size=self.params["inference.beam_size"])
        return decoder_output, decoding_res

    def _input_to_embedding_fn(self, x, time=None):
        """ Embeds the input symbols.

        Args:
            x: A 1/2-d Tensor to be embedded.
            time: An integer indicating the position. If `x` has shape
              [batch_size, timesteps], set None.

        Returns: A 2/3-d Tensor according to `x`.
        """
        with tf.variable_scope(self._input_modality.name):
            emb = self._input_modality.bottom(x, time)
        return emb

    def _target_to_embedding_fn(self, x, time=None):
        """ Embeds the input symbols.

        Args:
            x: A 1/2-d Tensor to be embedded.
            time: An integer indicating the position. If `x` has shape
              [batch_size, timesteps], set None.

        Returns: A 2/3-d Tensor according to `x`.
        """
        with tf.variable_scope(self._target_modality.name):
            emb = self._target_modality.targets_bottom(x, time)
        return emb

    def _outputs_to_logits_fn(self, outputs):
        """ Computes logits.

        Args:
            outputs: A Tensor with shape [..., dim]

        Returns: A Tensor with shape [..., vocab_size]
        """
        with tf.variable_scope(self._target_modality.name):
            logits = self._target_modality.top(outputs)
        return logits

    def _encode(self, input_fields):
        """ Calls encoder's encode method.

        Args:
            input_fields: A dictionary of placeholders.

        Returns: The results of encoding, an instance of `collections.namedtuple`
          from `Encoder.encode()`.
        """
        feature_ids = input_fields[Constants.FEATURE_IDS_NAME]
        feature_length = input_fields[Constants.FEATURE_LENGTH_NAME]

        if self.params["source.reverse"]:
            feature_ids = tf.reverse_sequence(
                input=feature_ids,
                seq_lengths=feature_length,
                batch_axis=0, seq_axis=1)
        features = self._input_to_embedding_fn(feature_ids)
        encoder_output = self._encoder.encode(features, feature_length)
        return encoder_output

    def _create_encoder(self):
        """ Creates encoder according model parameters.

        Returns: An instance of `Encoder`.
        """
        encoder_cls_name = self.params['encoder.class']
        if self.verbose:
            tf.logging.info("Creating ENCODER: {} for {}".format(encoder_cls_name, self.mode))
        encoder = eval(encoder_cls_name)(
            params=self.params['encoder.params'],
            mode=self.mode,
            name=encoder_cls_name.split(".")[-1],
            verbose=self.verbose)
        return encoder

    def _create_decoder(self):
        """ Creates decoder according model parameters.

        Returns: An instance of `Decoder`.
        """
        decoder_cls_name = self.params['decoder.class']
        if self.verbose:
            tf.logging.info("Creating DECODER: {}".format(decoder_cls_name))
        decoder = eval(decoder_cls_name)(
            self.params['decoder.params'], self.mode,
            name=decoder_cls_name.split(".")[-1], verbose=self.verbose)
        return decoder

    def _create_bridge(self):
        """ Creates bridge between encoder and decoder according
        to model parameters, initialized by `encoder_output`.

        Returns: An instance of `Bridge`.
        """
        encdec_bridge = eval(self.params["bridge.class"])(
            params=self.params["bridge.params"],
            mode=self.mode,
            verbose=self.verbose)
        return encdec_bridge

    def _pack_output(self,
                     encoder_output,
                     decoder_output,
                     decoding_result,
                     **kwargs):
        """ Packs model outputs.

        Args:
            encoder_output: An instance of `collections.namedtuple`
              from `Encoder.encode()`.
            decoder_output: An instance of `collections.namedtuple`
              whose element types are defined by `Decoder.output_dtype`
              property.
            decoding_result: A dict containing hypothesis, log
              probabilities, beam ids and decoding length if
              mode==INFER, else, a logits Tensor with shape
              [timesteps, batch_size, vocab_size].
            **kwargs: e.g. input fields.

        Returns: A dictionary containing inference status if mode==INFER,
         else a list with the first element be `loss`.
        """
        if self.mode == ModeKeys.TRAIN or self.mode == ModeKeys.EVAL:
            loss_sum, weight_sum = self._compute_loss(
                logits=decoding_result,  # [timesteps, batch_size, dim]
                label_ids=kwargs[Constants.LABEL_IDS_NAME],
                label_length=kwargs[Constants.LABEL_LENGTH_NAME])
        if self.mode == ModeKeys.TRAIN:
            return loss_sum, weight_sum

        attentions = dict()

        def get_attention(name, atts):
            if isinstance(atts, list):
                for idx, a in enumerate(atts):  # for multi-layer
                    attentions[name + str(idx)] = a
            else:
                attentions[name] = atts

        if hasattr(encoder_output, "encoder_self_attention"):
            # now it can be only MultiHeadAttention with shape [batch_size, num_heads, length_q, length_k]
            get_attention("encoder_self_attention", getattr(encoder_output, "encoder_self_attention"))
        if hasattr(decoder_output, "encoder_decoder_attention"):
            get_attention("encoder_decoder_attention", getattr(decoder_output, "encoder_decoder_attention"))
        if hasattr(decoder_output, "decoder_self_attention"):
            get_attention("decoder_self_attention", getattr(decoder_output, "decoder_self_attention"))

        if self.mode == ModeKeys.EVAL:
            return (loss_sum, weight_sum), attentions

        assert self.mode == ModeKeys.INFER
        predict_out = process_beam_predictions(
            decoding_result=decoding_result,
            beam_size=self.params["inference.beam_size"],
            alpha=self.params["inference.length_penalty"])
        predict_out["attentions"] = attentions
        predict_out["source"] = kwargs[Constants.FEATURE_IDS_NAME]
        return predict_out
