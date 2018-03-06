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

from njunmt.decoders.decoder import _embed_words, _compute_logits
from njunmt.utils.beam_search import stack_beam_size
from njunmt.utils.beam_search import BeamSearchStateSpec
from njunmt.utils.beam_search import gather_states
from njunmt.utils.beam_search import process_beam_predictions
from njunmt.utils.expert_utils import DecoderOutputRemover
from njunmt.utils.feedback import BeamFeedback
from njunmt.utils.constants import Constants


def dynamic_ensemble_decode(
        decoders,
        encoder_outputs,
        bridges,
        target_modalities,
        helper,
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
        target_modalities: A list of `Modality` instances.
        helper: An instance of `Feedback` that samples next symbols
          from logits.
        parallel_iterations: Argument passed to `tf.while_loop`.
        swap_memory: Argument passed to `tf.while_loop`.
        kwargs:

    Returns: The results of inference, an instance of `collections.namedtuple`
      whose element types are defined by `BeamSearchStateSpec`, indicating
      the status of beam search.
    """
    var_scope = tf.get_variable_scope()
    # Properly cache variable values inside the while_loop
    if var_scope.caching_device is None:
        var_scope.set_caching_device(lambda op: op.device)

    def _create_ta(d):
        return tf.TensorArray(
            dtype=d, clear_after_read=False,
            size=0, dynamic_size=True)

    decoder_output_removers = nest.map_structure(lambda dec: DecoderOutputRemover(
        dec.mode, dec.output_dtype._fields, dec.output_ignore_fields), decoders)

    # initialize first inputs (start of sentence) with shape [_batch*_beam,]
    initial_finished, initial_input_symbols = helper.init_symbols()
    initial_time = tf.constant(0, dtype=tf.int32)
    initial_inputs = nest.map_structure(
        lambda modality: _embed_words(modality, initial_input_symbols, initial_time),
        target_modalities)
    assert "beam_size" in kwargs
    beam_size = kwargs["beam_size"]
    initial_caches = []
    for dec, enc_out, bri in zip(decoders, encoder_outputs, bridges):
        with tf.variable_scope(dec.name):
            init_cache = dec.prepare(enc_out, bri, helper)  # prepare decoder
            init_cache = stack_beam_size(init_cache, beam_size)
            initial_caches.append(init_cache)

    initial_outputs_tas = nest.map_structure(
        lambda dec_out_rem, dec: nest.map_structure(
            _create_ta, dec_out_rem.apply(dec.output_dtype)),
        decoder_output_removers, decoders)

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
        outputs = []
        next_caches = []
        for dec, inp, cache in zip(decoders, inputs, caches):
            with tf.variable_scope(dec.name):
                out, next_cache = dec.step(inp, cache)
                outputs.append(out)
                next_caches.append(next_cache)
        next_outputs_tas = []
        for out_ta, out, rem in zip(outputs_tas, outputs, decoder_output_removers):
            ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                    out_ta, rem.apply(out))
            next_outputs_tas.append(ta)
        logits = []
        for dec, modality, out in zip(decoders, target_modalities, outputs):
            logits.append(_compute_logits(dec, modality, out))
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
        next_inputs = nest.map_structure(
            lambda modality: _embed_words(modality, next_input_symbols, time + 1),
            target_modalities)
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
                 weight_scheme,
                 inference_options):
        """ Initializes ensemble model parameters.

        Args:
            weight_scheme: A string, the ensemble weights. See
              `get_ensemble_weights()` for more details.
            inference_options: Contains beam_size, length_penalty
              and maximum_labels_length.
        """
        self._weight_scheme = weight_scheme
        self._beam_size = inference_options["beam_size"]
        self._length_penalty = inference_options["length_penalty"]
        self._maximum_labels_length = inference_options["maximum_labels_length"]

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

    def build(self, input_fields, base_models, vocab_target):
        """ Builds the ensemble model.

        Args:
            input_fields: A dictionary of placeholders.
            base_models: A list of `BaseSeq2Seq` instances.
            vocab_target: An instance of `Vocab`.

        Returns: A dictionary of inference status.
        """
        encoder_outputs = []
        encdec_bridges = []
        decoders = []
        target_modalities = []
        # prepare for decoding of each model
        for index, model in enumerate(base_models):
            with tf.variable_scope(
                            Constants.ENSEMBLE_VARNAME_PREFIX + str(index)):
                with tf.variable_scope(model.name):
                    input_modality, target_modality = model._create_modalities()
                    encoder = model._create_encoder()
                    encoder_output = model._encode(
                        encoder=encoder, input_modality=input_modality,
                        input_fields=input_fields)
                    bridge = model._create_bridge(encoder_output)
                    decoder = model._create_decoder()
                    vs_name = tf.get_variable_scope().name
                    decoder.name = os.path.join(vs_name, decoder.name)
                    target_modality.name = os.path.join(vs_name, target_modality.name)
                encoder_outputs.append(encoder_output)
                encdec_bridges.append(bridge)
                decoders.append(decoder)
                target_modalities.append(target_modality)

        helper = BeamFeedback(
            vocab=vocab_target,
            batch_size=tf.shape(input_fields[Constants.FEATURE_IDS_NAME])[0],
            maximum_labels_length=self._maximum_labels_length,
            beam_size=self._beam_size,
            alpha=self._length_penalty,
            ensemble_weight=self.get_ensemble_weights(len(base_models)))
        decoding_result = dynamic_ensemble_decode(
            decoders=decoders,
            encoder_outputs=encoder_outputs,
            bridges=encdec_bridges,
            target_modalities=target_modalities,
            helper=helper,
            beam_size=self._beam_size)
        predict_out = process_beam_predictions(
            decoding_result=decoding_result,
            beam_size=self._beam_size,
            alpha=self._length_penalty)
        predict_out["source"] = input_fields[Constants.FEATURE_IDS_NAME]
        return predict_out
