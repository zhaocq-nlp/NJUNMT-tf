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
""" Define a sequence-to-sequence model with attention. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from njunmt.utils.configurable import ModeKeys
from njunmt.models.base_seq2seq import BaseSeq2Seq
from njunmt.utils.global_names import GlobalNames
import tensorflow as tf


class AttentionSeq2Seq(BaseSeq2Seq):
    """ Define a sequence-to-sequence model with attention"""
    def __init__(self,
                 params,
                 mode,
                 vocab_source,
                 vocab_target,
                 name="attention_seq2seq",
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
        super(AttentionSeq2Seq, self).__init__(params=params, mode=mode,
                                               vocab_source=vocab_source,
                                               vocab_target=vocab_target,
                                               name=name, verbose=verbose)

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this model. """
        return {
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
            # "target.reverse": False,
            "inference.beam_size": 12,
            "inference.maximum_labels_length": 200,
            "inference.length_penalty": 0.0}

    def initializer(self):
        """ Returns the default initializer of the model scope.

        Returns: A `tf.initializer`.
        """
        dmodel = self.params["embedding.dim.target"]
        return tf.random_uniform_initializer(
            -dmodel ** -0.5, dmodel ** -0.5)

    def _pack_output(self,
                     encoder_output,
                     decoder_output,
                     infer_status,
                     target_modality,
                     **kwargs):
        """ Packs model outputs.

        Args:
            encoder_output: An instance of `collections.namedtuple`
              from `Encoder.encode()`.
            decoder_output: An instance of `collections.namedtuple`
              whose element types are defined by `Decoder.output_dtype`
              property.
            infer_status: An instance of `collections.namedtuple`
              whose element types are defined by `BeamSearchStateSpec`,
              indicating the status of beam search if mode==INFER, else,
              a logits Tensor with shape [timesteps, batch_size, vocab_size].
            target_modality: An instance of `Modality`.
            **kwargs:

        Returns: A dictionary containing inference status if mode==INFER,
         else a list with the first element be `loss`.
        """
        base_output = super(AttentionSeq2Seq, self)._pack_output(
            encoder_output, decoder_output, infer_status, target_modality, **kwargs)
        att = None
        if hasattr(decoder_output, "attention_scores"):
            att = decoder_output.attention_scores
            if self.params["source.reverse"]:
                att = tf.reverse_sequence(
                    input=decoder_output.attention_scores,  # [n_timesteps_trg, batch_size, n_timesteps_src]
                    seq_lengths=kwargs[GlobalNames.PH_FEATURE_IDS_NAME],
                    batch_axis=1, seq_axis=2)
        if att is not None:
            if self.mode == ModeKeys.INFER:
                base_output["attention_scores"] = att
            else:
                base_output = list(base_output)
                base_output += [att]
        return base_output
