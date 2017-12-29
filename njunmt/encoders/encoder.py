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
""" Base Decoder class. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod
from collections import namedtuple
from njunmt.utils.configurable import Configurable


class Encoder(Configurable):
    """ Base class for encoders. """

    def __init__(self, params, mode, name=None, verbose=True):
        """ Initializes the parameters of the encoder.

        Args:
            params: A dictionary of parameters to construct the
              encoder architecture.
            mode: A mode.
            name: The name of this encoder.
            verbose: Print encoder parameters if set True.
        """
        super(Encoder, self).__init__(
            params=params, mode=mode, verbose=verbose,
            name=name or self.__class__.__name__)
        # outputs: [batch_size, max_len, dim_hidden]
        # final_states: dict(), auxiliary states for decoder initializaiton
        # attention_values
        # attention_length
        self._encoder_output_tuple_type = namedtuple(
            "EncoderOutput",
            "outputs final_states attention_values attention_length")

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this encoder. """
        raise NotImplementedError

    @abstractmethod
    def encode(self, feature_ids, feature_length, input_modality, **kwargs):
        """ Encodes the inputs.

        Args:
            feature_ids: A Tensor, [batch_size, max_features_length].
            feature_length: A Tensor, [batch_size, ].
            input_modality: An instance of `Modality`.
            **kwargs:

        Returns: An instance of `collections.namedtuple`.
        """
        raise NotImplementedError
