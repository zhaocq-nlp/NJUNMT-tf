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
""" Base Decoder class and dynamic decode function. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod, abstractproperty
from tensorflow.python.util import nest
import tensorflow as tf

from njunmt.utils.constants import ModeKeys
from njunmt.utils.configurable import Configurable
from njunmt.utils.beam_search import stack_beam_size
from njunmt.utils.beam_search import gather_states
from njunmt.utils.beam_search import BeamSearchStateSpec
from njunmt.utils.expert_utils import DecoderOutputRemover


class Decoder(Configurable):
    """Base class for decoders. """

    def __init__(self, params, mode, name=None, verbose=True):
        """ Initializes the parameters of the decoder.

        Args:
            params: A dictionary of parameters to construct the
              decoder architecture.
            mode: A mode.
            name: The name of this decoder.
            verbose: Print decoder parameters if set True.
        """
        super(Decoder, self).__init__(
            params=params, mode=mode, verbose=verbose,
            name=name or self.__class__.__name__)

    @staticmethod
    def default_params():
        """ Returns a dictionary of default parameters of this decoder. """
        raise NotImplementedError

    @abstractproperty
    def output_dtype(self):
        """ Returns a `collections.namedtuple`,
        the definition of decoder output types. """
        raise NotImplementedError

    @abstractmethod
    def prepare(self, encoder_output, bridge, helper):
        """ Prepares for `step()` function.
        For example,
            1. initialize decoder hidden states (for RNN decoders);
            2. acquire attention information from `encoder_output`;
            3. pre-project the attention values if needed
            4. ...

        Args:
            encoder_output: An instance of `collections.namedtuple`
              from `Encoder.encode()`.
            bridge: An instance of `Bridge` that initializes the
              decoder states.
            helper: An instance of `Feedback` that samples next
              symbols from logits.
        Returns: A dict containing decoding states, pre-projected attention
          keys, attention values and attention length, and will be passed
          to `step()` function.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, decoder_input, cache):
        """ Decodes one step.

        Args:
            decoder_input: The decoder input for this timestep, an
              instance of `tf.Tensor`, [batch_size, dim_word].
            cache: A dict containing decoder RNN states at previous
              timestep, pre-projected attention keys, attention values
              and attention length.

        Returns: A tuple `(cur_decoder_outputs, cur_cache)` at this timestep.
          The `cur_decoder_outputs` must be an instance of `collections.namedtuple`
          whose element types are defined by `output_dtype` property. The
          `cur_cache` must have the same structure with `cache`.
        """
        raise NotImplementedError

    @abstractmethod
    def merge_top_features(self, decoder_output):
        """ Merges features of decoder top layers, as the input
        of softmax layer.

        Args:
            decoder_output: An instance of `collections.namedtuple`
              whose element types are defined by `output_dtype`
              property.

        Returns: A instance of `tf.Tensor`, as the input of
          softmax layer.
        """
        raise NotImplementedError

    @property
    def output_ignore_fields(self):
        """ Returns a list/tuple of strings. The loop in
        `dynamic_decode` function will not save these fields in
        `output_dtype` during inference, for the sake of reducing
        device memory.
        """
        return None

    def decode(self, encoder_output, bridge, helper,
               target_to_embedding_fn,
               outputs_to_logits_fn,
               **kwargs):
        """ Decodes one sample.

        Args:
            encoder_output: An instance of `collections.namedtuple`
              from `Encoder.encode()`.
            bridge: An instance of `Bridge` that initializes the
              decoder states.
            helper: An instance of `Feedback` that samples next
              symbols from logits.
            target_to_embedding_fn: A callable, converts target ids to
              embeddings.
            outputs_to_logits_fn: A callable, converts decoder outputs
              to logits.
            kwargs:

        Returns: A tuple `(decoder_output, decoder_status)`. The
          `decoder_output` is an instance of `collections.namedtuple`
          whose element types are defined by `output_dtype` property.
          For mode=INFER, the `decoder_status` is a dict containing
          hypothesis, log probabilities, beam ids and decoding length.
          For mode=TRAIN/EVAL, the `decoder_status` is a `tf.Tensor`
          indicating logits (computed by `target_modality`), of shape
          [timesteps, batch_size, vocab_size].
        """
        ret_val = dynamic_decode(
            decoder=self,
            encoder_output=encoder_output,
            bridge=bridge,
            helper=helper,
            target_to_embedding_fn=target_to_embedding_fn,
            outputs_to_logits_fn=outputs_to_logits_fn,
            **kwargs)
        if self.mode == ModeKeys.INFER:
            outputs, bs_results = ret_val
            return outputs, bs_results
        with tf.variable_scope(self.name):
            decoder_top_features = self.merge_top_features(ret_val)
        logits = outputs_to_logits_fn(decoder_top_features)
        return ret_val, logits


def _compute_logits(decoder, target_modality, decoder_output):
    """ Computes logits.

    Args:
        decoder: An instance of `Decoder.
        target_modality: An instance of `Modality`.
        decoder_output: An instance of `collections.namedtuple`
        whose element types are defined by `decoder.output_dtype`.

    Returns: A `tf.Tensor`.
    """
    with tf.variable_scope(decoder.name):
        decoder_top_features = decoder.merge_top_features(decoder_output)
    with tf.variable_scope(target_modality.name):
        logits = target_modality.top(decoder_top_features)
    return logits


def initialize_cache(
        decoding_states,
        attention_keys=None,
        memory=None,
        memory_bias=None):
    """ Creates a cache dict for tf.while_loop.

    Args:
        decoding_states: A Tensor or a structure of Tensors for decoding while loop.
        attention_keys: A Tensor. The attention keys for encoder-decoder attention.
        memory: A Tensor. The attention values for encoder-decoder attention.
        memory_bias: A Tensor. The attention bias for encoder-decoder attention.

    Returns: A dict.
    """
    cache = {"decoding_states": decoding_states}
    # encoder-related information (not influenced by beam search)
    if attention_keys is not None:
        cache["attention_keys"] = attention_keys
    if memory is not None:
        cache["memory"] = memory
    if memory_bias is not None:
        cache["memory_bias"] = memory_bias
    return cache


def _embed_words(target_modality, symbols, time):
    """ Embeds words into embeddings.

    Calls prepare() once and step() repeatedly on `Decoder` object.

    Args:
        target_modality: A `Modality` object.
        symbols: A `tf.Tensor` of 1-d, [batch_size, ].
        time: An integer or a scalar int32 tensor,
          indicating the position of this batch of symbols.

    Returns: A `tf.Tensor`, [batch_size, dimension].
    """
    with tf.variable_scope(target_modality.name):
        embeddings = target_modality.targets_bottom(symbols, time=time)
        return embeddings


def dynamic_decode(decoder,
                   encoder_output,
                   bridge,
                   helper,
                   target_to_embedding_fn,
                   outputs_to_logits_fn,
                   parallel_iterations=32,
                   swap_memory=False,
                   **kwargs):
    """ Performs dynamic decoding with `decoder`.

    Call `prepare()` once and `step()` repeatedly on the `Decoder` object.

    Args:
        decoder: An instance of `Decoder`.
        encoder_output: An instance of `collections.namedtuple`
          from `Encoder.encode()`.
        bridge: An instance of `Bridge` that initializes the
          decoder states.
        helper: An instance of `Feedback` that samples next
          symbols from logits.
        target_to_embedding_fn: A callable, converts target ids to
          embeddings.
        outputs_to_logits_fn: A callable, converts decoder outputs
          to logits.
        parallel_iterations: Argument passed to `tf.while_loop`.
        swap_memory: Argument passed to `tf.while_loop`.
        kwargs:

    Returns: A tuple `(decoder_output, decoder_status)` for
      decoder.mode=INFER.
      `decoder_output` for decoder.mode=TRAIN/INFER.
    """
    var_scope = tf.get_variable_scope()
    # Properly cache variable values inside the while_loop
    if var_scope.caching_device is None:
        var_scope.set_caching_device(lambda op: op.device)

    def _create_ta(d):
        return tf.TensorArray(
            dtype=d, clear_after_read=False,
            size=0, dynamic_size=True)

    decoder_output_remover = DecoderOutputRemover(
        decoder.mode, decoder.output_dtype._fields, decoder.output_ignore_fields)

    # initialize first inputs (start of sentence) with shape [_batch*_beam,]
    initial_finished, initial_input_symbols = helper.init_symbols()
    initial_time = tf.constant(0, dtype=tf.int32)
    initial_inputs = target_to_embedding_fn(initial_input_symbols, initial_time)

    with tf.variable_scope(decoder.name):
        initial_cache = decoder.prepare(encoder_output, bridge, helper)  # prepare decoder
        if decoder.mode == ModeKeys.INFER:
            assert "beam_size" in kwargs
            beam_size = kwargs["beam_size"]
            initial_cache = stack_beam_size(initial_cache, beam_size)

    initial_outputs_ta = nest.map_structure(
        _create_ta, decoder_output_remover.apply(decoder.output_dtype))

    def body_traininfer(time, inputs, cache, outputs_ta,
                        finished, *args):
        """Internal while_loop body.

        Args:
          time: scalar int32 Tensor.
          inputs: The inputs Tensor.
          cache: The decoder states.
          outputs_ta: structure of TensorArray.
          finished: A bool tensor (keeping track of what's finished).
          args: The log_probs, lengths, infer_status for mode==INFER.
        Returns:
          `(time + 1, next_inputs, next_cache, outputs_ta,
          next_finished, *args)`.
        """
        with tf.variable_scope(decoder.name):
            outputs, next_cache = decoder.step(inputs, cache)
        outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                        outputs_ta, decoder_output_remover.apply(outputs))
        inner_loop_vars = [time + 1, None, None, outputs_ta, None]
        sample_ids = None
        if decoder.mode == ModeKeys.INFER:
            log_probs, lengths = args[0], args[1]
            bs_stat_ta = args[2]
            predicted_ids = args[3]
            with tf.variable_scope(decoder.name):
                decoder_top_features = decoder.merge_top_features(outputs)
            logits = outputs_to_logits_fn(decoder_top_features)
            # sample next symbols
            sample_ids, beam_ids, next_log_probs, next_lengths \
                = helper.sample_symbols(logits, log_probs, finished, lengths, time=time)
            predicted_ids = gather_states(tf.reshape(predicted_ids, [-1, time + 1]), beam_ids)

            next_cache["decoding_states"] = gather_states(next_cache["decoding_states"], beam_ids)
            bs_stat = BeamSearchStateSpec(
                log_probs=next_log_probs,
                beam_ids=beam_ids)
            bs_stat_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                            bs_stat_ta, bs_stat)
            next_predicted_ids = tf.concat([predicted_ids, tf.expand_dims(sample_ids, axis=1)], axis=1)
            next_predicted_ids = tf.reshape(next_predicted_ids, [-1])
            next_predicted_ids.set_shape([None])
            inner_loop_vars.extend([next_log_probs, next_lengths, bs_stat_ta, next_predicted_ids])

        next_finished, next_input_symbols = helper.next_symbols(time=time, sample_ids=sample_ids)
        next_inputs = target_to_embedding_fn(next_input_symbols, time + 1)

        next_finished = tf.logical_or(next_finished, finished)
        inner_loop_vars[1] = next_inputs
        inner_loop_vars[2] = next_cache
        inner_loop_vars[4] = next_finished
        return inner_loop_vars

    loop_vars = [initial_time, initial_inputs, initial_cache,
                 initial_outputs_ta, initial_finished]

    if decoder.mode == ModeKeys.INFER:  # add inference-specific parameters
        initial_log_probs = tf.zeros_like(initial_input_symbols, dtype=tf.float32)
        initial_lengths = tf.zeros_like(initial_input_symbols, dtype=tf.int32)
        initial_bs_stat_ta = nest.map_structure(_create_ta, BeamSearchStateSpec.dtypes())
        # to process hypothesis
        initial_input_symbols.set_shape([None])
        loop_vars.extend([initial_log_probs, initial_lengths, initial_bs_stat_ta,
                          initial_input_symbols])

    res = tf.while_loop(
        lambda *args: tf.logical_not(tf.reduce_all(args[4])),
        body_traininfer,
        loop_vars=loop_vars,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory)

    final_outputs_ta = res[3]
    final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)

    if decoder.mode == ModeKeys.INFER:
        timesteps = res[0] + 1
        log_probs, length, bs_stat, predicted_ids = res[-4:]
        final_bs_stat = nest.map_structure(lambda ta: ta.stack(), bs_stat)
        return final_outputs, \
               {"beam_ids": final_bs_stat.beam_ids,
                "log_probs": final_bs_stat.log_probs,
                "decoding_length": length,
                "hypothesis": tf.reshape(predicted_ids, [-1, timesteps])[:, 1:]}

    return final_outputs
