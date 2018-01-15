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
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest
from abc import abstractmethod
from seq2seq.utils.configurable import Configurable
from seq2seq.utils.algebra_ops import fflayer


def _final_state(x, direction):
    if len(x) == 1 or direction == "forward":
        return x[0][-1]
    elif direction == "backward":
        return x[1][-1]
    else:
        raise ValueError("Unrecognized type of direction: %s" % direction)


class Bridge(Configurable):
    """ base bridge class"""

    def __init__(self, params, encoder_output, mode):
        super(Bridge, self).__init__(params, mode)
        self.encoder_output = encoder_output
        self.batch_size = tf.shape(nest.flatten(self.encoder_output.final_state)[0])[0]

    @staticmethod
    def default_params():
        return {}

    @abstractmethod
    def _create(self, decoder_state_size, **kwargs):
        """
        :param decoder_state_size: decoder state size
        :param kwargs: maybe has "beam_size"
        :return:
        """
        raise NotImplementedError

    def __call__(self, decoder_state_size, **kwargs):
        return self._create(decoder_state_size, **kwargs)


class ZeroBridge(Bridge):
    """ A bridge that does not pass any information between encoder and decoder
    and sets the initial decoder state to 0. The input function is not modified"""

    def __init__(self, params, encoder_output, mode):
        super(ZeroBridge, self).__init__(params, encoder_output, mode)
        tf.logging.info("Using ZeroBridge...Initialize decoder state with all zero vectors...")

    @staticmethod
    def default_params():
        return {}

    def _create(self, decoder_state_size, **kwargs):
        """
        if decoder_state_size if int/LSTMStateTuple, return int/LSTMStateTuple with shape [batch_size, hidden_size]
        else if decoder_state_size is tuple, return a tuple with each element to be int/LSTMStateTuple type
        :param decoder_state_size: decoder state size
        :param name:
        :return:
        """
        if "beam_size" in kwargs:
            return rnn_cell_impl._zero_state_tensors(decoder_state_size,
                                                     self.batch_size * kwargs["beam_size"], tf.float32)
        return rnn_cell_impl._zero_state_tensors(decoder_state_size, self.batch_size, tf.float32)


class PassThroughBridge(Bridge):
    """ pass the encoder final state to decoder"""

    def __init__(self, params, encoder_output, mode):
        super(PassThroughBridge, self).__init__(params, encoder_output, mode)
        tf.logging.info("Using PassThroughBridge...Pass the last encoder state to decoder...")

    @staticmethod
    def default_params():
        # "forward" or "backward"
        return {"direction": "backward"}

    def _create(self, decoder_state_size, **kwargs):
        batch_size = self.batch_size
        # of type LSTMStateTuple
        enc_final_state = _final_state(self.encoder_output.final_state, direction=self.params["direction"])
        if "beam_size" in kwargs:
            beam_size = kwargs["beam_size"]

            def _stack_beam_size(x):
                if isinstance(x, tf.Tensor):
                    if x.get_shape().ndims == 3:
                        # tile: [batch_size, max_len_seq * beam_size, num_units]
                        #    reshape: [batch_size * beam_size, max_len_seq, num_units]
                        final_shape = [beam_size * batch_size, -1, x.get_shape().as_list()[-1]]
                        return tf.reshape(tf.tile(x, [1, beam_size, 1]), final_shape)
                    elif x.get_shape().ndims == 1:
                        return tf.reshape(
                            tf.transpose(tf.tile([x], [beam_size, 1])), [-1])
                    elif x.get_shape().ndims == 2:
                        # tile: [batch_size, num_units * beam_size]
                        #    reshape: [batch_size * beam_size, num_units]
                        final_shape = [batch_size * beam_size, x.get_shape().as_list()[-1]]
                        return tf.reshape(tf.tile(x, [1, beam_size]), final_shape)
                    else:
                        raise NotImplementedError("Not implemented the capability for ndims=%d" % x.get_shape().ndims)
                else:
                    raise ValueError("Unrecognized state type: %s" % type(x))

            enc_final_state = nest.pack_sequence_as(enc_final_state,
                                                    nest.map_structure(_stack_beam_size,
                                                                       nest.flatten(enc_final_state)))
        if nest.is_sequence(decoder_state_size):
            return tuple([enc_final_state for _ in decoder_state_size])
        return enc_final_state


class InitialStateBridge(Bridge):
    """ initialize decoder state with projection of encoder output
    or final state"""

    def __init__(self, params, encoder_output, mode):
        super(InitialStateBridge, self).__init__(params, encoder_output, mode)
        tf.logging.info("Using InitialStateBridge...Initialze decoder state with projection of encoder %s..."
                        % self.params["bridge_input"])
        self._activation = self.params["activation"]

    @staticmethod
    def default_params():
        return {
            # "final state" or "outputs"
            "bridge_input": "outputs",
            "activation": tf.tanh
        }

    def _create(self, decoder_state_size, **kwargs):
        if not hasattr(self.encoder_output, self.params["bridge_input"]):
            raise ValueError("encoder output has not attribute: %s, "
                             "only final_state and outputs available" % self.params["bridge_input"])
        if self.params["bridge_input"] == "outputs":
            # [batch_size, max_time, num_units]
            context = self.encoder_output.outputs
            mask = tf.sequence_mask(
                lengths=tf.to_int32(self.encoder_output.attention_length),
                maxlen=tf.shape(context)[1],
                dtype=tf.float32)
            # [batch_size, num_units]
            bridge_input = tf.truediv(tf.reduce_sum(context * tf.expand_dims(mask, 2), axis=1),
                                      tf.expand_dims(
                                          tf.to_float(self.encoder_output.attention_length), 1))
        elif self.params["bridge_input"] == "final_state":
            bridge_input = nest.flatten(self.encoder_output.final_state)
            bridge_input = tf.concat(bridge_input, 1)
        else:
            raise ValueError("Unrecognized value of bridge_input: %s, "
                             "should be outputs or final_state" % self.params["bridge_input"])
        state_size_splits = nest.flatten(decoder_state_size)
        total_decoder_state_size = sum(state_size_splits)
        # [batch_size, total_decoder_state_size]
        init_state = fflayer(inputs=bridge_input,
                             output_size=total_decoder_state_size,
                             activation=self._activation,
                             name="init_state_trans")
        if "beam_size" in kwargs:
            beam_size = kwargs["beam_size"]
            batch_size = self.batch_size

            def _stack_beam_size(x):
                if isinstance(x, tf.Tensor) and x.get_shape().ndims == 2:
                    # tile: [batch_size, num_units * beam_size]
                    #    reshape: [batch_size * beam_size, num_units]
                    final_shape = [batch_size * beam_size, x.get_shape().as_list()[-1]]
                    return tf.reshape(tf.tile(x, [1, beam_size]), final_shape)
                else:
                    raise ValueError("Unrecognized state type: %s or ndims: %d" % type(x), x.get_shape().ndims)

            init_state = nest.map_structure(_stack_beam_size, init_state)
        init_state = nest.pack_sequence_as(decoder_state_size,
                                           tf.split(init_state, state_size_splits, axis=1))
        return init_state


class VariableBridge(Bridge):
    """ learn the initial states of the decoder itself

    """

    def __init__(self, params, encoder_output, mode):
        super(VariableBridge, self).__init__(params, encoder_output, mode)
        tf.logging.info("Using VariableBridge...try to learn the initial state of decoder...")

    @staticmethod
    def default_params():
        return {}

    def _create(self, decoder_state_size, **kwargs):
        name = kwargs["name"] if "name" in kwargs else None
        state_size_splits = nest.flatten(decoder_state_size)
        total_decoder_state_size = sum(state_size_splits)
        with tf.variable_scope(name or "init_state"):
            init_state_total = tf.get_variable("init_states", shape=(total_decoder_state_size,),
                                               dtype=tf.float32, initializer=tf.zeros_initializer)
        if "beam_size" in kwargs:
            beam_size = kwargs["beam_size"]
            init_state_total = tf.tile([init_state_total], [beam_size * self.batch_size, 1])
        else:
            init_state_total = tf.tile([init_state_total], [self.batch_size, 1])
        init_state = nest.pack_sequence_as(decoder_state_size,
                                           tf.split(init_state_total, state_size_splits, axis=1))
        return init_state
