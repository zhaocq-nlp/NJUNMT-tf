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
""" Define RNN-based encoders. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from njunmt.encoders.encoder import Encoder
from njunmt.utils.rnn_cell_utils import get_multilayer_rnn_cells


class StackBidirectionalRNNEncoder(Encoder):
    """ Define stacked bidirectional RNN encoder. """

    def __init__(self, params, mode, name=None, verbose=True):
        """ Initializes the parameters of the encoder.

        Args:
            params: A dictionary of parameters to construct the
              encoder architecture.
            mode: A mode.
            name: The name of this encoder.
            verbose: Print encoder parameters if set True.
        """
        super(StackBidirectionalRNNEncoder, self).__init__(params, mode, name, verbose)
        self._cells_fw = get_multilayer_rnn_cells(**self.params['rnn_cell'])
        self._cells_bw = get_multilayer_rnn_cells(**self.params['rnn_cell'])

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this encoder. """
        return {
            "rnn_cell": {
                "cell_class": "LSTMCell",
                "cell_params": { },
                "dropout_input_keep_prob": 1.0,
                "dropout_state_keep_prob": 1.0,
                "num_layers": 1
            }
        }

    def encode(self, feature_ids, feature_length, input_modality, **kwargs):
        """ Encodes the inputs via a stacked bi-directional RNN.

        Args:
            feature_ids: A Tensor, [batch_size, max_features_length].
            feature_length: A Tensor, [batch_size, ].
            input_modality: An instance of `Modality`.
            **kwargs:

        Returns: An instance of `collections.namedtuple`.
        """
        with tf.variable_scope(input_modality.name):
            inputs = input_modality.bottom(feature_ids)
        scope = self.name
        if "scope" in kwargs:
            scope = kwargs.pop("scope")
        # outputs: [batch_size, max_time, layers_output]
        #   layers_output = size_of_fw + size_of_bw
        # the returned states:
        #   `tuple` type which has only one item, because we use MultiRNN cell for multiple cells
        outputs, states_fw, states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=[self._cells_fw],
            cells_bw=[self._cells_bw],
            inputs=inputs,
            sequence_length=feature_length,
            dtype=tf.float32,
            scope=scope,
            **kwargs)

        # because we use MultiRNNCell, unpack the top tuple structure
        states_fw = states_fw[0]
        states_bw = states_bw[0]

        return self._encoder_output_tuple_type(
            outputs=outputs,
            final_states={
                "forward": states_fw[-1],
                "backward": states_bw[-1]},
            attention_values=outputs,
            attention_length=feature_length)


class UnidirectionalRNNEncoder(Encoder):
    """ Define a unidirectional RNN encoder. """

    def __init__(self, params, mode, name=None, verbose=True):
        """ Initializes the parameters of the encoder.

        Args:
            params: A dictionary of parameters to construct the
              encoder architecture.
            mode: A mode.
            name: The name of this encoder.
            verbose: Print encoder parameters if set True.
        """
        super(UnidirectionalRNNEncoder, self).__init__(params, mode, name, verbose)
        self._cells_fw = get_multilayer_rnn_cells(**self.params['rnn_cell'])

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this encoder. """
        return {
            "rnn_cell": {
                "cell_class": "LSTMCell",
                "cell_params": {},
                "dropout_input_keep_prob": 1.0,
                "dropout_state_keep_prob": 1.0,
                "num_layers": 1
            }
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
        scope = self.name
        if "scope" in kwargs:
            scope = kwargs.pop("scope")
        # outputs: [batch_size, max_time, num_units_of_hidden]
        outputs, states = tf.nn.dynamic_rnn(
            cell=self._cells_fw,
            inputs=inputs,
            sequence_length=feature_length,
            dtype=tf.float32,
            scope=scope,
            **kwargs)

        return self._encoder_output_tuple_type(
            outputs=outputs,
            final_statest=states[-1],
            attention_values=outputs,
            attention_length=feature_length)
