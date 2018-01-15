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

from abc import abstractmethod
import tensorflow as tf
from tensorflow.python.util import nest
from seq2seq.utils.configurable import Configurable


class Decoder(Configurable):
    def __init__(self, params, mode):
        super(Decoder, self).__init__(params, mode)
        self._batch_size = None
        self._beam_size = 1

    @staticmethod
    def default_params():
        return {}

    @property
    def output_size(self):
        return self._output_size

    @property
    def output_dtype(self):
        return self._output_dtype

    @property
    def beam_size(self):
        return self._beam_size

    @property
    def batch_size(self):
        if self._batch_size is None:
            raise ValueError("_batch_size in Decoder is not assigned a value")
        return self._batch_size * self.beam_size

    @abstractmethod
    def initialize(self):
        raise NotImplementedError

    @abstractmethod
    def finalize(self, final_outputs, final_state):
        raise NotImplementedError

    @abstractmethod
    def step(self, time_, inputs, state, aux_inputs):
        raise NotImplementedError

    @abstractmethod
    def compute_logit(self, decoder_output, scope=None):
        raise NotImplementedError

    @abstractmethod
    def _setup(self, encoder_output, bridge, helper):
        raise NotImplementedError

    def decode(self, encoder_output, bridge, helper, scope=None):
        """
        decode the sentence
        :param encoder_output: encoder output named tuple
        :param bridge: `Bridge` sub instance
        :param helper: `FeedBack` sub instance
        :param scope:
        :return: decoder output
        """
        with tf.variable_scope(scope or self.__class__.__name__):
            self._setup(encoder_output, bridge, helper)

            scope = tf.get_variable_scope()
            scope.set_initializer(tf.random_uniform_initializer(
                -self.params["init_scale"],
                self.params["init_scale"]))

            outputs, final_state = dynamic_decode(decoder=self, scope=scope)
        return self.finalize(final_outputs=outputs, final_state=final_state)


def dynamic_decode(decoder,
                   maximum_iterations=None,
                   parallel_iterations=32,
                   swap_memory=False,
                   scope=None):
    """Perform dynamic decoding with `decoder`.

    Calls initialize() once and step() repeatedly on the Decoder object.

    :param decoder: A `Decoder` instance.
    :param maximum_iterations: `int32` scalar, maximum allowed number of decoding
         steps.  Default is `None` (decode until the decoder is fully done).
    :param parallel_iterations: Argument passed to `tf.while_loop`.
    :param swap_memory: Argument passed to `tf.while_loop`.
    :param scope: Optional variable scope to use.
    :return: `(final_outputs, final_sequence_lengths)`.
    """
    default_vs = tf.get_variable_scope()
    with tf.variable_scope(scope or default_vs) as var_scope:
        # Properly cache variable values inside the while_loop
        if var_scope.caching_device is None:
            var_scope.set_caching_device(lambda op: op.device)

        if maximum_iterations is not None:
            maximum_iterations = tf.convert_to_tensor(
                maximum_iterations, dtype=tf.int32, name="maximum_iterations")
            if maximum_iterations.get_shape().ndims != 0:
                raise ValueError("maximum_iterations must be a scalar")

        initial_finished, initial_inputs, initial_state, \
            init_aux_inputs = decoder.initialize()

        if maximum_iterations is not None:
            initial_finished = tf.logical_or(
                initial_finished, 0 >= maximum_iterations)
        initial_time = tf.constant(0, dtype=tf.int32)

        def _shape(batch_size, from_shape):
            if not isinstance(from_shape, tf.TensorShape):
                return tf.TensorShape(None)
            else:
                batch_size = tf.contrib.util.constant_value(
                    tf.convert_to_tensor(
                        batch_size, name="batch_size"))
                return tf.TensorShape([batch_size]).concatenate(from_shape)

        def _create_ta(s, d):
            return tf.TensorArray(
                dtype=d,
                size=0,
                dynamic_size=True,
                element_shape=_shape(decoder.batch_size, s))

        initial_outputs_ta = nest.map_structure(_create_ta, decoder.output_size,
                                                decoder.output_dtype)

        def condition(unused_time, unused_outputs_ta, unused_state, unused_inputs,
                      unused_aux_inputs, finished):
            return tf.logical_not(tf.reduce_all(finished))

        def body(time, outputs_ta, state, inputs, aux_inputs, finished):
            """Internal while_loop body.
            Args:
              time: scalar int32 tensor.
              outputs_ta: structure of TensorArray.
              state: (structure of) state tensors and TensorArrays.
              inputs: (structure of) input tensors.
              finished: bool tensor (keeping track of what's finished).
            Returns:
              `(time + 1, outputs_ta, next_state, next_inputs, next_finished,
                next_sequence_lengths)`.
              ```
            """
            (next_outputs, decoder_finished, next_inputs, decoder_state,
             next_aux_inputs) = decoder.step(time, inputs, state, aux_inputs)
            next_finished = tf.logical_or(decoder_finished, finished)
            if maximum_iterations is not None:
                next_finished = tf.logical_or(
                    next_finished, time + 1 >= maximum_iterations)

            nest.assert_same_structure(state, decoder_state)
            nest.assert_same_structure(outputs_ta, next_outputs)
            nest.assert_same_structure(inputs, next_inputs)
            nest.assert_same_structure(aux_inputs, next_aux_inputs)

            emit = next_outputs
            next_state = decoder_state

            outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                            outputs_ta, emit)
            next_inputs.set_shape(inputs.get_shape())
            return time + 1, outputs_ta, next_state, next_inputs, \
                   next_aux_inputs, next_finished

        res = tf.while_loop(
            condition,
            body,
            loop_vars=[
                initial_time, initial_outputs_ta, initial_state, initial_inputs,
                init_aux_inputs, initial_finished,
            ],
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)

        final_outputs_ta = res[1]
        final_state = res[2]

        final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)

    return final_outputs, final_state

