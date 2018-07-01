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
import tensorflow as tf
from tensorflow import gfile

from njunmt.inference.attention import postprocess_attention
from njunmt.inference.attention import select_attention_sample_by_sample
from njunmt.inference.attention import pack_batch_attention_dict
from njunmt.inference.attention import dump_attentions
from njunmt.tools.tokenizeChinese import to_chinese_char
from njunmt.utils.expert_utils import repeat_n_times


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


def evaluate(sess, loss_op, eval_data):
    """ Evaluates data by loss.

    Args:
        sess: `tf.Session`.
        loss_op: Tensorflow operation, computing the loss.
        eval_data: An iterable instance that each element
          is a packed feeding dictionary for `sess`.

    Returns: Total loss averaged by number of data samples.
    """
    losses = []
    weights = []
    total_loss = 0.
    total_weight = 0.
    for data in eval_data:
        parallels = data["feed_dict"].pop("parallels")
        avail = sum(numpy.array(parallels) > 0)
        loss = _evaluate(sess, data["feed_dict"], loss_op[:avail])
        data["feed_dict"]["parallels"] = parallels
        total_loss += sum([_l[0] for _l in loss])
        total_weight += sum([_l[1] for _l in loss])
    loss = total_loss / total_weight
    return loss


def evaluate_with_attention(
        sess,
        loss_op,
        eval_data,
        vocab_source,
        vocab_target,
        attention_op=None,
        output_filename_prefix=None):
    """ Evaluates data by loss.

    Args:
        sess: `tf.Session`.
        loss_op: Tensorflow operation, computing the loss.
        eval_data: An iterable instance that each element
          is a packed feeding dictionary for `sess`.
        vocab_source: A `Vocab` instance for source side feature map.
        vocab_target: A `Vocab` instance for target side feature map.
        attention_op: Tensorflow operation for output attention.
        output_filename_prefix: A string.

    Returns: Total loss averaged by number of data samples.
    """
    losses = 0.
    weights = 0.
    num_of_samples = 0
    attentions = {}
    for data in eval_data:
        _n_samples = len(data["feature_ids"])
        parallels = data["feed_dict"].pop("parallels")
        avail = sum(numpy.array(parallels) > 0)
        if attention_op is None:
            loss = _evaluate(sess, data["feed_dict"], loss_op[:avail])
        else:
            loss, atts = _evaluate(sess, data["feed_dict"],
                                   [loss_op[:avail], attention_op[:avail]])
            ss_strs = [vocab_source.convert_to_wordlist(ss, bpe_decoding=False)
                       for ss in data["feature_ids"]]
            tt_strs = [vocab_target.convert_to_wordlist(
                tt, bpe_decoding=False, reverse_seq=False)
                       for tt in data["label_ids"]]
            _attentions = sum(repeat_n_times(avail, select_attention_sample_by_sample,
                                             atts), [])
            attentions.update(pack_batch_attention_dict(
                num_of_samples, ss_strs, tt_strs, _attentions))
        data["feed_dict"]["parallels"] = parallels
        losses += sum([_l[0] for _l in loss])
        weights += sum([_l[1] for _l in loss])
        num_of_samples += _n_samples
    loss = losses / weights
    if attention_op is not None:
        dump_attentions(output_filename_prefix, attentions)
    return loss


def _infer(
        sess,
        feed_dict,
        prediction_op,
        batch_size,
        top_k=1,
        output_attention=False):
    """ Infers a batch of samples with beam search.

    Args:
        sess: `tf.Session`
        feed_dict: A dictionary of feeding data.
        prediction_op: A list of Tensorflow operation for inference.
        batch_size: The batch size.
        top_k: An integer, number of predicted sequences will be
          returned.
        output_attention: Whether to output attention.

    Returns: A tuple `(predicted_sequences, attention_scores)`.
      The `predicted_sequences` is a list of hypothesis with
      approx [`top_k` * `batch_sze`, sequence_length].
      The `attention_scores` is None if there is no attention
      related information in `prediction_op`.
    """
    parallels = feed_dict.pop("parallels")
    avail = sum(numpy.array(parallels) > 0)
    extract_keys = ["sorted_hypothesis", "sorted_scores"]
    if output_attention:
        assert top_k == 1, (
            "`output_attention` flag now only accepts `tok_k`=1.")
        extract_keys.extend(["sorted_argidx", "attentions", "beam_ids"])
    brief_pred_op = dict(zip(
        extract_keys,
        repeat_n_times(
            avail,
            lambda dd: tuple([dd[k] for k in extract_keys]),
            prediction_op[:avail])))
    predict_out = sess.run(brief_pred_op, feed_dict=feed_dict)
    feed_dict["parallels"] = parallels
    total_samples = sum(
        repeat_n_times(avail,
                       lambda p: p.shape[0],
                       predict_out["sorted_hypothesis"]))
    beam_size = total_samples // batch_size

    def _post_process_hypo(pred, score, **kwargs):
        _num_samples = pred.shape[0]
        _batch_size = _num_samples // beam_size
        batch_beam_pos = numpy.tile(numpy.arange(_batch_size) * beam_size, [beam_size, 1]).transpose()
        batch_beam_topk_add = numpy.tile(numpy.arange(top_k), [batch_beam_pos.shape[0], 1])
        batch_beam_pos = numpy.reshape(
            batch_beam_pos[:, :top_k] + batch_beam_topk_add, -1)
        if output_attention:
            atts = postprocess_attention(
                beam_ids=kwargs["beam_ids"],
                attention_dict=kwargs["attentions"],
                gather_idx=kwargs["sorted_argidx"][batch_beam_pos])
            return pred[batch_beam_pos, :].tolist(), score[batch_beam_pos], atts
        # [_batch * _beam, timesteps] => [_batch * top_k, timesteps]
        return pred[batch_beam_pos, :].tolist(), score[batch_beam_pos], []

    hypothesises, scores, attentions = repeat_n_times(
        avail,
        _post_process_hypo,
        predict_out["sorted_hypothesis"],
        predict_out["sorted_scores"],
        beam_ids=predict_out.get("beam_ids", None),
        attentions=predict_out.get("attentions", None),
        sorted_argidx=predict_out.get("sorted_argidx", None))
    hypothesis = sum(hypothesises, [])
    score = numpy.concatenate(scores, axis=0)
    attention = sum(attentions, [])
    return hypothesis, score, attention


def infer(
        sess,
        prediction_op,
        infer_data,
        output,
        vocab_source,
        vocab_target,
        delimiter=" ",
        output_attention=False,
        tokenize_output=False,
        verbose=True):
    """ Infers data and save the prediction results.

    Args:
        sess: `tf.Session`.
        prediction_op: Tensorflow operation for inference.
        infer_data: An iterable instance that each element
          is a packed feeding dictionary for `sess`.
        output: Output file name, `str`.
        vocab_source: A `Vocab` instance for source side feature map.
        vocab_target: A `Vocab` instance for target side feature map.
        alpha: A scalar number, length penalty rate. If not provided
          or < 0, simply average each beam by length of predicted
          sequence.
        delimiter: The delimiter of output token sequence.
        output_attention: Whether to output attention information.
        tokenize_output: Whether to split words into characters
          (only for Chinese).
        verbose: Print inference information if set True.

    Returns: A tuple `(sources, hypothesis)`, two lists of
      strings.
    """
    attentions = dict()
    hypothesis = []
    scores = []
    sources = []
    cnt = 0
    for data in infer_data:
        source_tokens = [vocab_source.convert_to_wordlist(
            x, bpe_decoding=False, reverse_seq=False)
                         for x in data["feature_ids"]]
        x_str = [delimiter.join(x) for x in source_tokens]
        prediction, score, att = _infer(
            sess=sess,
            feed_dict=data["feed_dict"],
            prediction_op=prediction_op,
            batch_size=len(x_str),
            top_k=1,
            output_attention=output_attention)

        sources.extend(x_str)
        scores.append(score)
        hypothesis.extend([delimiter.join(vocab_target.convert_to_wordlist(prediction[sample_idx]))
                           for sample_idx in range(len(prediction))])
        if output_attention and att is not None:
            candidate_tokens = [vocab_target.convert_to_wordlist(
                prediction[idx], bpe_decoding=False, reverse_seq=False)
                                for idx in range(len(x_str))]

            attentions.update(pack_batch_attention_dict(
                cnt, source_tokens, candidate_tokens, att))
        cnt += len(x_str)
        if verbose:
            tf.logging.info(cnt)
    if tokenize_output:
        hypothesis = to_chinese_char(hypothesis)
    if output:
        with gfile.GFile(output, "w") as fw:
            fw.write("\n".join(hypothesis) + "\n")
    if output_attention:
        dump_attentions(output, attentions)
    return sources, hypothesis, numpy.concatenate(scores, axis=0)
