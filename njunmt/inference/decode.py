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
""" Common inference and evaluation functions. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import json
import os
import random
import string
import re
import tensorflow as tf
from tensorflow import gfile

from njunmt.inference.attention import process_attention_output
from njunmt.inference.attention import pack_batch_attention_dict
from njunmt.utils.global_names import GlobalNames
from njunmt.utils.misc import padding_batch_data


def _evaluate(
        sess,
        feed_dict,
        eval_op):
    """ Evaluates one batch of feeding data.

    Args:
        sess: `tf.Session`.
        feed_dict: A dictionary of feeding data.
        eval_op: Tensorflow operation.

    Returns: Results of `eval_op`.

    """
    return sess.run(eval_op, feed_dict=feed_dict)


def evaluate_sentences(
        sources,
        targets,
        sess,
        input_fields,
        eval_op,
        vocab_source,
        vocab_target,
        n_words_src=-1,
        n_words_trg=-1):
    """ Evaluates a list of sentences.

    Args:
        sources: A list of strings.
        targets: A list of strings.
        sess: `tf.Session`.
        input_fields: The dictionary of placeholders.
        eval_op: Tensorflow operation.
        vocab_source: A `Vocab` instance for source side feature map.
        vocab_target: A `Vocab` instance for target side feature map.
        n_words_src: An integer number. If provided and > 0, source side
          token id that exceed this value will be mapped into UNK id.
        n_words_trg: An integer number. If provided and > 0, target side
          token id that exceed this value will be mapped into UNK id.

    Returns: Results of `eval_op`.
    """
    sources = [vocab_source.convert_to_idlist(
        re.split(r"\s*", snt.strip()), n_words_src) for snt in sources]
    targets = [vocab_target.convert_to_idlist(
        re.split(r"\s*", snt.strip()), n_words_trg) for snt in targets]
    ph_x = input_fields[GlobalNames.PH_FEATURE_IDS_NAME]
    ph_x_len = input_fields[GlobalNames.PH_FEATURE_LENGTH_NAME]
    ph_y = input_fields[GlobalNames.PH_LABEL_IDS_NAME]
    ph_y_len = input_fields[GlobalNames.PH_LABEL_LENGTH_NAME]
    x, len_x = padding_batch_data(sources, vocab_source.eos_id)
    y, len_y = padding_batch_data(targets, vocab_target.eos_id)
    feed_dict = {ph_x: x, ph_x_len: len_x,
                 ph_y: y, ph_y_len: len_y}
    return _evaluate(sess, feed_dict, eval_op)


def evaluate(
        sess,
        eval_op,
        feeding_data):
    """ Evaluates data by loss.

    Args:
        sess: `tf.Session`.
        eval_op: Tensorflow operation, computing the loss.
        feeding_data: An iterable instance that each element
          is a packed feeding dictionary for `sess`.

    Returns: Total loss averaged by number of data samples.
    """
    losses = 0.
    total_size = 0
    for num_data, feed_dict in feeding_data:
        loss = _evaluate(sess, feed_dict, eval_op)
        losses += loss * float(num_data)
        total_size += num_data
    loss = losses / float(total_size)
    return loss


def _infer(
        sess,
        feed_dict,
        prediction_op,
        batch_size,
        alpha=None,
        top_k=1,
        output_attention=False):
    """ Infers a batch of samples with beam search.

    Args:
        sess: `tf.Session`
        feed_dict: A dictionary of feeding data.
        prediction_op: Tensorflow operation for inference.
        batch_size: An integer, the number of data samples.
        alpha: A scalar number, length penalty rate. If not provided
          or < 0, simply average each beam by length of predicted
          sequence.
        top_k: An integer, number of predicted sequences will be
          returned.
        output_attention: Whether to output attention.

    Returns: A tuple `(predicted_sequences, attention_scores)`.
      The `predicted_sequences` is an ndarray of shape
      [`top_k` * `batch_sze`, max_sequence_length].
      The `attention_scores` is None if there is no attention
      related information in `prediction_op`.
    """
    if output_attention:
        predict_out = sess.run(prediction_op,
                               feed_dict=feed_dict)
    else:
        att_op = prediction_op.pop("attentions")
        predict_out = sess.run(prediction_op,
                               feed_dict=feed_dict)
        prediction_op["attentions"] = att_op
    predicted_ids = predict_out["predicted_ids"]  # [n_timesteps_trg, batch_size * beam_size]
    beam_ids = predict_out["beam_ids"]  # [n_timesteps_trg, batch_size * beam_size]
    sequence_lengths = predict_out["sequence_lengths"]  # [n_timesteps_trg, batch_size * beam_size]
    log_probs = predict_out["log_probs"]  # [n_timesteps_trg, batch_size * beam_size]

    gathered_pred_ids = numpy.zeros_like(beam_ids)  # [n_timesteps_trg, batch_size * beam_size]
    for idx in range(beam_ids.shape[0]):
        gathered_pred_ids = gathered_pred_ids[:, beam_ids[idx]]
        gathered_pred_ids[idx, :] = predicted_ids[idx]
    gathered_pred_ids = gathered_pred_ids.transpose()  # [batch_size * beam_size, n_timesteps_trg]

    lengths = numpy.array(sequence_lengths[-1], dtype=numpy.float32)
    if alpha is None or alpha < 0.:
        penalty = lengths
    else:
        penalty = ((5.0 + lengths) / 6.0) ** alpha

    scores = log_probs[-1] / penalty
    beam_size = scores.shape[0] // batch_size
    scores = scores.reshape([-1, beam_size])  # [batch_size, beam_size]

    argidx = numpy.argsort(scores, axis=1)[:, ::-1]  # descending order: [batch_size, beam_size]
    argidx += numpy.tile(numpy.arange(batch_size) * beam_size, [beam_size, 1]).transpose()
    argidx = numpy.reshape(argidx[:, :top_k], -1)
    if output_attention:
        attentions = process_attention_output(predict_out, argidx)
        return gathered_pred_ids[argidx, :], attentions
    return gathered_pred_ids[argidx, :], None


def infer_sentences(
        sources,
        sess,
        input_fields,
        prediction_op,
        vocab_source,
        alpha=None,
        top_k=1,
        n_words_src=-1):
    """ Infers a list of sentences.

    Args:
        sources: A list of strings.
        sess: `tf.Session`.
        input_fields: The dictionary of placeholders.
        prediction_op: Tensorflow operation for inference.
        vocab_source: A `Vocab` instance for source side feature map.
        alpha: A scalar number, length penalty rate. If not provided
          or < 0, simply average each beam by length of predicted
          sequence.
        top_k: An integer, number of predicted sequences will be
          returned.
        n_words_src: An integer number. If provided and > 0, source side
          token id that exceed this value will be mapped into UNK id.

    Returns: A tuple `(predicted_sequences, attention_scores)`.
      The `predicted_sequences` is an ndarray of shape
      [`top_k`, max_sequence_length].
      The `attention_scores` is None if there is no attention
      related information in `prediction_op`.
    """
    sources = [vocab_source.convert_to_idlist(
        re.split(r"\s*", snt.strip()), n_words_src) for snt in sources]
    ph_x = input_fields[GlobalNames.PH_FEATURE_IDS_NAME]
    ph_x_len = input_fields[GlobalNames.PH_FEATURE_LENGTH_NAME]
    x, len_x = padding_batch_data(sources, vocab_source.eos_id)
    feed_dict = {ph_x: x, ph_x_len: len_x}
    return _infer(sess, feed_dict, prediction_op, len(sources), alpha, top_k)


def infer(
        sess,
        prediction_op,
        feeding_data,
        output,
        vocab_target,
        alpha=None,
        delimiter=" ",
        output_attention=False,
        tokenize_output=False,
        tokenize_script="./njunmt/tools/tokenizeChinese.py",
        verbose=True):
    """ Infers data and save the prediction results.

    Args:
        sess: `tf.Session`.
        prediction_op: Tensorflow operation for inference.
        feeding_data: An iterable instance that each element
          is a packed feeding dictionary for `sess`.
        output: Output file name, `str`.
        vocab_target: A `Vocab` instance for target side feature map.
        alpha: A scalar number, length penalty rate. If not provided
          or < 0, simply average each beam by length of predicted
          sequence.
        delimiter: The delimiter of output token sequence.
        output_attention: Whether to output attention information.
        tokenize_output: Whether to split words into characters
          (only for Chinese).
        tokenize_script: The script for `tokenize_output`.
        verbose: Print inference information if set True.

    Returns: A tuple `(sample_src, sample_trg)`, two lists of
      strings. Sample from `feeding_data`.
    """
    attentions = dict()
    samples_src = []
    samples_trg = []
    with gfile.GFile(output, "w") as fw:
        cnt = 0
        for x_str, x_len, feeding_batch in feeding_data:
            prediction, att = _infer(sess, feeding_batch, prediction_op,
                                     len(x_str), alpha=alpha, top_k=1,
                                     output_attention=output_attention)
            y_str = [delimiter.join(vocab_target.convert_to_wordlist(prediction[sample_idx]))
                     for sample_idx in range(prediction.shape[0])]
            fw.write('\n'.join(y_str) + "\n")
            # random sample
            if random.random() < 0.3 and len(samples_src) < 5:
                for sample_idx in range(len(x_str)):
                    samples_src.append(x_str[sample_idx])
                    samples_trg.append(y_str[sample_idx])
                    if len(samples_src) >= 5:
                        break

            # output attention
            if output_attention and att is not None:
                source_tokens = [x.strip().split() for x in x_str]
                candidate_tokens = [vocab_target.convert_to_wordlist(
                    prediction[idx, :], bpe_decoding=False, reverse_seq=False)
                                    for idx in range(len(x_str))]

                attentions.update(pack_batch_attention_dict(
                    cnt, source_tokens, candidate_tokens, att))
            cnt += len(x_str)
            if verbose:
                tf.logging.info(cnt)
    if tokenize_output:
        tmp_output_file = output + ''.join((''.join(
            random.sample(string.digits + string.ascii_letters, 10))).split())
        os.system("python %s %s %s" %
                  (tokenize_script, output, tmp_output_file))
        os.system("mv %s %s" % (tmp_output_file, output))
    if output_attention:
        with gfile.GFile(output + ".attention", "wb") as f:
            f.write(json.dumps(attentions).encode("utf-8"))
    return samples_src, samples_trg
