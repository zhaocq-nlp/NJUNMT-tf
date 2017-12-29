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
""" Define RNNEncoder-RNNDecoder bridge classes. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.framework import tensor_util
from tensorflow.contrib.framework.python.framework.tensor_util import with_same_shape

from njunmt.layers.common_layers import fflayer
from njunmt.utils.configurable import Configurable


def assert_state_is_compatible(expected_state, state):
    """Asserts that states are compatible.

    Args:
        expected_state: The reference state.
        state: The state that must be compatible with :obj:`expected_state`.

    Raises:
      ValueError: if the states are incompatible.
    """
    # Check structure compatibility.
    nest.assert_same_structure(expected_state, state)

    # Check shape compatibility.
    expected_state_flat = nest.flatten(expected_state)
    state_flat = nest.flatten(state)

    for x, y in zip(expected_state_flat, state_flat):
        if tensor_util.is_tensor(x):
            with_same_shape(x, y)


def _final_state(x, direction):
    """ Acquires final states.

    Args:
        x: A Tensor/LSTMStateTuple or a dictionary of Tensors/LSTMStateTuples.
        direction: The key for `x` if `x` is a dictionary.

    Returns: A Tensor or a LSTMStateTuple, according to x.

    Raises:
        ValueError: if the type of x is not mentioned above, or if `direction`
          is not a valid key of `x`, when `x` is a dictionary.
    """
    if isinstance(x, tf.Tensor) or isinstance(x, rnn_cell_impl.LSTMStateTuple):
        return x
    elif isinstance(x, dict):
        try:
            ret = x[direction]
        except KeyError:
            raise ValueError(
                "Unrecognized type of direction: {}".format(direction))
        return ret
    else:
        raise ValueError(
            "Unrecognized type of direction: {} "
            "or unknow type of final_states: {}".format(direction, type(x)))


def _final_states(x):
    """ Get the final states if `x` is a dictionary.

    Args:
        x: A dictionary of states or others.

    Returns: `x` if `x` is not a dictionary, otherwise, a list of values in `x`.
    """
    if isinstance(x, dict):
        ret = []
        for k, v in x.items():
            ret.append(v)
        return ret
    return x


class Bridge(Configurable):
    """ Define base bridge class. """

    def __init__(self, params, encoder_output, mode, name=None, verbose=True):
        """ Initializes bridge parameters.

        Args:
            arams: A dictionary of parameters.
            encoder_output: An instance of `collections.namedtuple`
              from `Encoder.encode()`.
            mode: A mode.
            name: The name of this bridge.
            verbose: Print bridge parameters if set True.
        """
        super(Bridge, self).__init__(
            params=params, mode=mode, verbose=verbose,
            name=name or self.__class__.__name__)
        self.encoder_output = encoder_output
        self.batch_size = tf.shape(encoder_output.attention_length)[0]

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this bridge. """
        return {}

    @abstractmethod
    def _create(self, decoder_state_size, **kwargs):
        """ Creates decoder's initial RNN states according to
        `decoder_state_size`.

        Args:
            decoder_state_size: RNN decoder state size.
            **kwargs:

        Returns: The decoder states with the structure determined
          by `decoder_state_size`.
        """
        raise NotImplementedError

    def __call__(self, decoder_state_size, **kwargs):
        """ Calls `_create()` function.

        Args:
            decoder_state_size: RNN decoder state size.
            **kwargs:

        Returns: The decoder states with the structure determined
          by `decoder_state_size`.
        """
        with tf.variable_scope(self.name):
            return self._create(decoder_state_size, **kwargs)


class ZeroBridge(Bridge):
    """ Define a bridge that does not pass any information between
    encoder and decoder, and sets the initial decoder state to 0."""

    def __init__(self, params, encoder_output, mode, name=None, verbose=True):
        """ Initializes bridge parameters.

        Args:
            arams: A dictionary of parameters.
            encoder_output: An instance of `collections.namedtuple`
              from `Encoder.encode()`.
            mode: A mode.
            name: The name of this bridge.
            verbose: Print bridge parameters if set True.
        """
        super(ZeroBridge, self).__init__(
            params=params, encoder_output=encoder_output,
            mode=mode, name=name, verbose=verbose)
        if self.verbose:
            tf.logging.info("Using ZeroBridge. Initialize decoder state with all zero vectors.")

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this bridge. """
        return {}

    def _create(self, decoder_state_size, **kwargs):
        """ Creates decoder's initial RNN states according to
        `decoder_state_size`.

        If `decoder_state_size` is int/LSTMStateTuple(int, int), return Tensor
        with shape [batch_size, int] or LSTMStateTuple([batch_size, int], [batch_size, int]).
        If `decoder_state_size` is a tuple of int/LSTMStateTupe, return a tuple
        whose elements' structure match the `decoder_state_size` respectively.
        Args:
            decoder_state_size: RNN decoder state size.
            **kwargs:

        Returns: The decoder states with the structure determined
          by `decoder_state_size`.
        """
        batch_size = self.batch_size
        return rnn_cell_impl._zero_state_tensors(
            decoder_state_size, batch_size, tf.float32)


class PassThroughBridge(Bridge):
    """ Define a bridge that passes the encoder's final state to decoder. """

    def __init__(self, params, encoder_output, mode, name=None, verbose=True):
        """ Initializes bridge parameters.

        Args:
            arams: A dictionary of parameters.
            encoder_output: An instance of `collections.namedtuple`
              from `Encoder.encode()`.
            mode: A mode.
            name: The name of this bridge.
            verbose: Print bridge parameters if set True.
        """
        super(PassThroughBridge, self).__init__(
            params=params, encoder_output=encoder_output,
            mode=mode, name=name, verbose=verbose)
        if self.verbose:
            tf.logging.info("Using PassThroughBridge. Pass the last encoder state to decoder.")

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this bridge. """
        # "forward" or "backward"
        return {"direction": "backward"}

    def _create(self, decoder_state_size, **kwargs):
        """ Creates decoder's initial RNN states according to
        `decoder_state_size`.

        Passes the final state of encoder to each layer in decoder.
        Args:
            decoder_state_size: RNN decoder state size.
            **kwargs:

        Returns: The decoder states with the structure determined
          by `decoder_state_size`.

        Raises:
            ValueError: if the structure of encoder RNN state does not
              have the same structure of decoder RNN state.
        """
        batch_size = self.batch_size
        # of type LSTMStateTuple
        enc_final_state = _final_state(
            self.encoder_output.final_states, direction=self.params["direction"])
        assert_state_is_compatible(rnn_cell_impl._zero_state_tensors(
            decoder_state_size[0],
            batch_size, tf.float32), enc_final_state)
        if nest.is_sequence(decoder_state_size):
            return tuple([enc_final_state for _ in decoder_state_size])
        return enc_final_state


class InitialStateBridge(Bridge):
    """ Define a bridge that initializes decoder state with projection
    of encoder output or final state"""

    def __init__(self, params, encoder_output, mode, name=None, verbose=True):
        """ Initializes bridge parameters.

        Args:
            arams: A dictionary of parameters.
            encoder_output: An instance of `collections.namedtuple`
              from `Encoder.encode()`.
            mode: A mode.
            name: The name of this bridge.
            verbose: Print bridge parameters if set True.
        """
        super(InitialStateBridge, self).__init__(
            params=params, encoder_output=encoder_output,
            mode=mode, name=name, verbose=verbose)
        if self.verbose:
            tf.logging.info("Using InitialStateBridge. Initialze decoder state with projection of encoder {}."
                            .format(self.params["bridge_input"]))
        self._activation = self.params["activation"]

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this bridge. """
        return {
            # "final_states" or "outputs"
            "bridge_input": "outputs",
            "activation": tf.tanh
        }

    def _create(self, decoder_state_size, **kwargs):
        """ Creates decoder's initial RNN states according to
        `decoder_state_size`.

        Do linear transformations to encoder output/state and map the
        structure to `decoder_state_size`.
        If params[`bridge_input`] == "output", first average the encoder
        output tensor over timesteps.
        Args:
            decoder_state_size: RNN decoder state size.
            **kwargs:

        Returns: The decoder states with the structure determined
          by `decoder_state_size`.

        Raises:
            ValueError: if `encoder_output` has no attribute named
              params[`bridge_input`].
        """
        if not hasattr(self.encoder_output, self.params["bridge_input"]):
            raise ValueError("encoder output has not attribute: {}, "
                             "only final_state and outputs available"
                             .format(self.params["bridge_input"]))
        if self.params["bridge_input"] == "outputs":
            # [batch_size, max_time, num_units]
            context = self.encoder_output.outputs
            mask = tf.sequence_mask(
                lengths=tf.to_int32(self.encoder_output.attention_length),
                maxlen=tf.shape(context)[1],
                dtype=tf.float32)
            # [batch_size, num_units]
            bridge_input = tf.truediv(
                tf.reduce_sum(context * tf.expand_dims(mask, 2), axis=1),
                tf.expand_dims(
                    tf.to_float(self.encoder_output.attention_length), 1))
        elif self.params["bridge_input"] == "final_states":
            bridge_input = nest.flatten(_final_states(self.encoder_output.final_states))
            bridge_input = tf.concat(bridge_input, 1)
        else:
            raise ValueError("Unrecognized value of bridge_input: {}, "
                             "should be outputs or final_state".format(self.params["bridge_input"]))
        state_size_splits = nest.flatten(decoder_state_size)
        total_decoder_state_size = sum(state_size_splits)
        # [batch_size, total_decoder_state_size]
        init_state = fflayer(inputs=bridge_input,
                             output_size=total_decoder_state_size,
                             activation=self._activation,
                             name="init_state_trans")
        init_state = nest.pack_sequence_as(
            decoder_state_size,
            tf.split(init_state, state_size_splits, axis=1))
        return init_state


class VariableBridge(Bridge):
    """ Define a bridge that learns the initial states of the
    decoder automatically. """

    def __init__(self, params, encoder_output, mode, name=None, verbose=True):
        """ Initializes bridge parameters.

        Args:
            arams: A dictionary of parameters.
            encoder_output: An instance of `collections.namedtuple`
              from `Encoder.encode()`.
            mode: A mode.
            name: The name of this bridge.
            verbose: Print bridge parameters if set True.
        """
        super(VariableBridge, self).__init__(
            params=params, encoder_output=encoder_output,
            mode=mode, name=name, verbose=verbose)
        if self.verbose:
            tf.logging.info("Using VariableBridge. Try to learn the initial state of decoder.")

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this bridge. """
        return {}

    def _create(self, decoder_state_size, **kwargs):
        """ Creates decoder's initial RNN states according to
        `decoder_state_size`.

        Creates a tf variable and passes to decoder.
        Args:
            decoder_state_size: RNN decoder state size.
            **kwargs:

        Returns: The decoder states with the structure determined
          by `decoder_state_size`.
        """
        name = kwargs["name"] if "name" in kwargs else None
        state_size_splits = nest.flatten(decoder_state_size)
        total_decoder_state_size = sum(state_size_splits)
        with tf.variable_scope(name or "init_state"):
            init_state_total = tf.get_variable(
                name="init_states", shape=(total_decoder_state_size,),
                dtype=tf.float32, initializer=tf.zeros_initializer)
        init_state_total = tf.tile([init_state_total], [self.batch_size, 1])
        init_state = nest.pack_sequence_as(
            decoder_state_size,
            tf.split(init_state_total, state_size_splits, axis=1))
        return init_state
