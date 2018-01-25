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

import copy

from njunmt.models.sequence_to_sequence import SequenceToSequence


class AttentionSeq2Seq(SequenceToSequence):
    """ Define a sequence-to-sequence model with attention.
    It is now reserved for old versions."""

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
        assert self.params["encoder.class"] == "njunmt.encoders.rnn_encoder.StackBidirectionalRNNEncoder", (
            "AttentionSeq2Seq must use StackBidirectionalRNNEncoder.")
        assert self.params["decoder.class"] == "njunmt.decoders.rnn_decoder.CondAttentionDecoder", (
            "AttentionSeq2Seq must use CondAttentionDecoder.")

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this model. """
        attseq_default_params = copy.deepcopy(SequenceToSequence.default_params())
        attseq_default_params["initializer"] = "random_uniform"
        attseq_default_params["encoder.class"] = "njunmt.encoders.rnn_encoder.StackBidirectionalRNNEncoder"
        attseq_default_params["decoder.class"] = "njunmt.decoders.rnn_decoder.CondAttentionDecoder"
        return attseq_default_params

