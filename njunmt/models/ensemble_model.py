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
""" Implement model ensemble."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.python.util import nest

from njunmt.utils.beam_search import stack_beam_size
from njunmt.utils.beam_search import BeamSearchStateSpec
from njunmt.utils.beam_search import gather_states
from njunmt.utils.beam_search import process_beam_predictions
from njunmt.utils.expert_utils import DecoderOutputRemover
from njunmt.utils.expert_utils import repeat_n_times
from njunmt.utils.feedback import BeamFeedback
from njunmt.utils.constants import Constants


def dynamic_ensemble_decode(
        decoders,
        encoder_outputs,
        bridges,
        helper,
        target_to_embedding_fns,
        outputs_to_logits_fns,
        parallel_iterations=32,
        swap_memory=False,
        **kwargs):
    """ Performs dynamic decoding with `decoders`.

    Calls prepare() once and step() repeatedly on `Decoder` object.

    Args:
        decoders: A list of `Decoder` instances.
        encoder_outputs: A list of `collections.namedtuple`s from each
          corresponding `Encoder.encode()`.
        bridges: A list of `Bridge` instances or Nones.
        helper: An instance of `Feedback` that samples next symbols
          from logits.
        target_to_embedding_fns: A list of callables, converts target ids to
          embeddings.
        outputs_to_logits_fns: A list of callables, converts decoder outputs
          to logits.
        parallel_iterations: Argument passed to `tf.while_loop`.
        swap_memory: Argument passed to `tf.while_loop`.
        kwargs:

    Returns: The results of inference, an instance of `collections.namedtuple`
      whose element types are defined by `BeamSearchStateSpec`, indicating
      the status of beam search.
    """
    num_models = len(decoders)
    var_scope = tf.get_variable_scope()
    # Properly cache variable values inside the while_loop
    if var_scope.caching_device is None:
        var_scope.set_caching_device(lambda op: op.device)

    def _create_ta(d):
        return tf.TensorArray(
            dtype=d, clear_after_read=False,
            size=0, dynamic_size=True)

    decoder_output_removers = repeat_n_times(
        num_models, lambda dec: DecoderOutputRemover(
            dec.mode, dec.output_dtype._fields, dec.output_ignore_fields), decoders)

    # initialize first inputs (start of sentence) with shape [_batch*_beam,]
    initial_finished, initial_input_symbols = helper.init_symbols()
    initial_time = tf.constant(0, dtype=tf.int32)
    initial_inputs = repeat_n_times(
        num_models, target_to_embedding_fns,
        initial_input_symbols, initial_time)

    assert "beam_size" in kwargs
    beam_size = kwargs["beam_size"]

    def _create_cache(_decoder, _encoder_output, _bridge):
        with tf.variable_scope(_decoder.name):
            _init_cache = _decoder.prepare(_encoder_output, _bridge, helper)
            _init_cache = stack_beam_size(_init_cache, beam_size)
        return _init_cache

    initial_caches = repeat_n_times(
        num_models, _create_cache,
        decoders, encoder_outputs, bridges)

    initial_outputs_tas = [nest.map_structure(
        _create_ta, _decoder_output_remover.apply(_decoder.output_dtype))
                           for _decoder_output_remover, _decoder in zip(decoder_output_removers, decoders)]

    def body_infer(time, inputs, caches, outputs_tas, finished,
                   log_probs, lengths, bs_stat_ta, predicted_ids):
        """Internal while_loop body.

        Args:
          time: Scalar int32 Tensor.
          inputs: A list of inputs Tensors.
          caches: A dict of decoder states.
          outputs_tas: A list of TensorArrays.
          finished: A bool tensor (keeping track of what's finished).
          log_probs: The log probability Tensor.
          lengths: The decoding length Tensor.
          bs_stat_ta: structure of TensorArray.
          predicted_ids: A Tensor.

        Returns:
          `(time + 1, next_inputs, next_caches, next_outputs_tas,
          next_finished, next_log_probs, next_lengths, next_infer_status_ta)`.
        """

        # step decoder
        def _decoding(_decoder, _input, _cache, _decoder_output_remover,
                      _outputs_ta, _outputs_to_logits_fn):
            with tf.variable_scope(_decoder.name):
                _output, _next_cache = _decoder.step(_input, _cache)
                _decoder_top_features = _decoder.merge_top_features(_output)
            _ta = nest.map_structure(lambda _ta_ms, _output_ms: _ta_ms.write(time, _output_ms),
                                     _outputs_ta, _decoder_output_remover.apply(_output))
            _logit = _outputs_to_logits_fn(_decoder_top_features)
            return _output, _next_cache, _ta, _logit

        outputs, next_caches, next_outputs_tas, logits = repeat_n_times(
            num_models, _decoding,
            decoders, inputs, caches, decoder_output_removers,
            outputs_tas, outputs_to_logits_fns)

        # sample next symbols
        sample_ids, beam_ids, next_log_probs, next_lengths \
            = helper.sample_symbols(logits, log_probs, finished, lengths, time=time)

        for c in next_caches:
            c["decoding_states"] = gather_states(c["decoding_states"], beam_ids)

        infer_status = BeamSearchStateSpec(
            log_probs=next_log_probs,
            beam_ids=beam_ids)
        bs_stat_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                        bs_stat_ta, infer_status)
        predicted_ids = gather_states(tf.reshape(predicted_ids, [-1, time + 1]), beam_ids)
        next_predicted_ids = tf.concat([predicted_ids, tf.expand_dims(sample_ids, axis=1)], axis=1)
        next_predicted_ids = tf.reshape(next_predicted_ids, [-1])
        next_predicted_ids.set_shape([None])
        next_finished, next_input_symbols = helper.next_symbols(time=time, sample_ids=sample_ids)
        next_inputs = repeat_n_times(num_models, target_to_embedding_fns,
                                     next_input_symbols, time + 1)
        next_finished = tf.logical_or(next_finished, finished)

        return time + 1, next_inputs, next_caches, next_outputs_tas, \
               next_finished, next_log_probs, next_lengths, bs_stat_ta, \
               next_predicted_ids

    initial_log_probs = tf.zeros_like(initial_input_symbols, dtype=tf.float32)
    initial_lengths = tf.zeros_like(initial_input_symbols, dtype=tf.int32)
    initial_bs_stat_ta = nest.map_structure(_create_ta, BeamSearchStateSpec.dtypes())
    initial_input_symbols.set_shape([None])
    loop_vars = [initial_time, initial_inputs, initial_caches,
                 initial_outputs_tas, initial_finished,
                 # infer vars
                 initial_log_probs, initial_lengths, initial_bs_stat_ta,
                 initial_input_symbols]

    res = tf.while_loop(
        lambda *args: tf.logical_not(tf.reduce_all(args[4])),
        body_infer,
        loop_vars=loop_vars,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory)

    timesteps = res[0] + 1
    log_probs, length, bs_stat, predicted_ids = res[-4:]
    final_bs_stat = nest.map_structure(lambda ta: ta.stack(), bs_stat)
    return {"beam_ids": final_bs_stat.beam_ids,
            "log_probs": final_bs_stat.log_probs,
            "decoding_length": length,
            "hypothesis": tf.reshape(predicted_ids, [-1, timesteps])[:, 1:]}


class EnsembleModel(object):
    """ Define the model ensemble wrapper class. """

    def __init__(self,
                 vocab_target,
                 base_models,
                 weight_scheme,
                 inference_options):
        """ Initializes ensemble model parameters.

        Args:
            vocab_target: A `Vocab` instance.
            base_models: A list of `SequenceToSequence` instances.
            weight_scheme: A string, the ensemble weights. See
              `get_ensemble_weights()` for more details.
            inference_options: Contains beam_size, length_penalty
              and maximum_labels_length.
        """
        self._vocab_target = vocab_target
        self._base_models = base_models
        self._weight_scheme = weight_scheme
        self._beam_size = inference_options["beam_size"]
        self._length_penalty = inference_options["length_penalty"]
        self._maximum_labels_length = inference_options["maximum_labels_length"]
        # update model components' names
        for model in self._base_models:
            model._decoder.name = os.path.join(model.name, model._decoder.name)
            model._encoder.name = os.path.join(model.name, model._encoder.name)
            model._input_modality.name = os.path.join(model.name, model._input_modality.name)
            model._target_modality.name = os.path.join(model.name, model._target_modality.name)
            model._encoder_decoder_bridge.name = os.path.join(model.name, model._encoder_decoder_bridge.name)

    def get_ensemble_weights(self, num_models):
        """ Creates ensemble weights from `weight_scheme`.

        Now, only weight_scheme="average" is available.

        Args:
            num_models: The number of single models.

        Returns: A list of floats. The size of it is `num_models`.

        Raises:
            NotImplementedError: if `weight_scheme` != "average".
        """
        if self._weight_scheme == "average":
            return [1.0 / float(num_models)] * int(num_models)
        # TODO can also directly process weights, like "0.1,0.1"
        raise NotImplementedError("This weight scheme is not implemented: {}."
                                  .format(self._weight_scheme))

    def build(self, input_fields):
        """ Builds the ensemble model.

        Args:
            input_fields: A dict of placeholders.

        Returns: A dictionary of inference status.
        """
        encoder_outputs = []
        # prepare for decoding of each model
        for index, model in enumerate(self._base_models):
            encoder_output = model._encode(input_fields=input_fields)
            encoder_outputs.append(encoder_output)

        helper = BeamFeedback(
            vocab=self._vocab_target,
            batch_size=tf.shape(input_fields[Constants.FEATURE_IDS_NAME])[0],
            maximum_labels_length=self._maximum_labels_length,
            beam_size=self._beam_size,
            alpha=self._length_penalty,
            ensemble_weight=self.get_ensemble_weights(len(self._base_models)))

        decoders, bridges, target_to_emb_fns, outputs_to_logits_fns = \
            repeat_n_times(
                len(self._base_models),
                lambda m: (m._decoder, m._encoder_decoder_bridge, m._target_to_embedding_fn, m._outputs_to_logits_fn),
                self._base_models)

        decoding_result = dynamic_ensemble_decode(
            decoders=decoders,
            encoder_outputs=encoder_outputs,
            bridges=bridges,
            helper=helper,
            target_to_embedding_fns=target_to_emb_fns,
            outputs_to_logits_fns=outputs_to_logits_fns,
            beam_size=self._beam_size)
        predict_out = process_beam_predictions(
            decoding_result=decoding_result,
            beam_size=self._beam_size,
            alpha=self._length_penalty)
        predict_out["source"] = input_fields[Constants.FEATURE_IDS_NAME]
        return predict_out
