# Copyright 2017 ZhaoChengqi, zhaocq@nlp.nju.edu.cn, Natural Language Processing Group, Nanjing University.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import namedtuple
from seq2seq.encoders.encoder import Encoder
from seq2seq.utils.rnn_cell_utils import get_multilayer_rnn_cells


class StackBidirectionalRNNEncoder(Encoder):
    """
    Stacked Bidirectional RNN Encoder
    """

    def __init__(self, params, mode):
        """
        Constructor for bidirectional rnn encoder
        :param params:
        :param mode:
        :param scope:
        :param kwargs:
        """
        super(StackBidirectionalRNNEncoder, self).__init__(params=params, mode=mode)
        self.cells_fw = get_multilayer_rnn_cells(**self.params['rnn_cell'])
        self.cells_bw = get_multilayer_rnn_cells(**self.params['rnn_cell'])
        self.encoder_output_tuple_type = namedtuple(
            "EncoderOutput",
            "outputs final_state attention_values attention_length")

    @staticmethod
    def default_params():
        return {
            "rnn_cell": {
                "cell_class": "LSTMCell",
                "cell_params": {
                    "num_units": 1024,
                },
                "dropout_input_keep_prob": 1.0,
                "dropout_state_keep_prob": 1.0,
                "num_layers": 1
            },
            "init_scale": 0.04
        }

    def encode(self, inputs, sequence_length, scope=None, **kwargs):
        """
        bidirectional rnn, to encode the input
        :param inputs:
        :param sequence_length:
        :param kwargs:
        :return:
        """
        with tf.variable_scope(scope or self.__class__.__name__) as vs:
            vs.set_initializer(tf.random_uniform_initializer(
                -self.params["init_scale"],
                self.params["init_scale"]))

            # outputs: [batch_size, max_time, layers_output]
            ##  layers_output = size_of_fw + size_of_bw
            # the returned states:
            #   `tuple` type which has only one item, because we use MultiRNN cell for multiple cells
            outputs, states_fw, states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw=[self.cells_fw],
                cells_bw=[self.cells_bw],
                inputs=inputs,
                sequence_length=sequence_length,
                dtype=tf.float32,
                scope=scope,
                **kwargs)

        return self.encoder_output_tuple_type(
            outputs=outputs,
            final_state=(states_fw[-1], states_bw[-1]),
            attention_values=outputs,
            attention_length=sequence_length)


class UnidirectionalRNNEncoder(Encoder):
    """
    Unidirectional RNN Encoder
    """

    def __init__(self, params, mode):
        """
        Constructor for bidirectional rnn encoder
        :param params:
        :param mode:
        :param scope:
        :param kwargs:
        """
        super(UnidirectionalRNNEncoder, self).__init__(params=params, mode=mode)
        self.cells_fw = get_multilayer_rnn_cells(**self.params['rnn_cell'])
        self.encoder_output_tuple_type = namedtuple(
            "EncoderOutput",
            "outputs final_state attention_values attention_length")

    @staticmethod
    def default_params():
        return {
            "rnn_cell": {
                "cell_class": "LSTMCell",
                "cell_params": {
                    "num_units": 1024,
                },
                "dropout_input_keep_prob": 1.0,
                "dropout_state_keep_prob": 1.0,
                "num_layers": 1
            },
            "init_scale": 0.04
        }

    def encode(self, inputs, sequence_length, scope=None, **kwargs):
        '''
        bidirectional rnn, to encode the input
        :param inputs:
        :param sequence_length:
        :param kwargs:
        :return:
        '''
        with tf.variable_scope(scope or self.__class__.__name__) as vs:
            vs.set_initializer(tf.random_uniform_initializer(
                -self.params["init_scale"],
                self.params["init_scale"]))
            # outputs: [batch_size, max_time, num_units_of_hidden]
            outputs, states = tf.nn.dynamic_rnn(
                cell=self.cells_fw,
                inputs=inputs,
                sequence_length=sequence_length,
                dtype=tf.float32,
                scope=scope,
                **kwargs)

        return self.encoder_output_tuple_type(
            outputs=outputs,
            final_state=tuple([states]),
            attention_values=outputs,
            attention_length=sequence_length)
