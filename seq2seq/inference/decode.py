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

import numpy
import json
import os
import random
import string

from seq2seq.utils.global_names import GlobalNames
from tensorflow import gfile
import tensorflow as tf


def evaluate(
        sess,
        input_fields,
        eval_op,
        evaldata_iterator):
    ph_x = input_fields[GlobalNames.PH_SOURCE_SEQIDS_NAME]
    ph_x_len = input_fields[GlobalNames.PH_SOURCE_SEQLENGTH_NAME]
    ph_y = input_fields[GlobalNames.PH_TARGET_SEQIDS_NAME]
    ph_y_len = input_fields[GlobalNames.PH_TARGET_SEQLENGTH_NAME]

    losses = 0.
    total_size = 0
    for (x, len_x), (y, len_y) in evaldata_iterator:
        feed_dict = {ph_x: x, ph_x_len: len_x,
                     ph_y: y, ph_y_len: len_y}
        loss = sess.run(eval_op, feed_dict=feed_dict)
        losses += loss * float(len(len_x))
        total_size += len(len_x)
    loss = losses / float(total_size)
    return loss


def infer(
        sess,
        input_fields,
        prediction_op,
        testdata_iterator,
        output,
        vocab_target,
        delimiter=" ",
        output_attention=False,
        tokenize_output=False,
        tokenize_script="./seq2seq/tools/tokenizeChinese.py",
        verbose=True):
    """ do inference

    :param sess:
    :param input_fields:
    :param prediction_op:
    :param testdata_iterator: `TestTextIterator`
    :param output:
    :param vocab_target:
    :param delimiter:
    :param output_attention:
    :param tokenize_output:
    :param tokenize_script:
    :param verbose:
    :return: samples
    """
    ph_x = input_fields[GlobalNames.PH_SOURCE_SEQIDS_NAME]
    ph_x_len = input_fields[GlobalNames.PH_SOURCE_SEQLENGTH_NAME]
    attentions = dict()
    samples_src = []
    samples_trg = []
    with gfile.GFile(output, "w") as fw:
        cnt = 0
        for x_str, (x, lengths_x) in testdata_iterator:
            predict_out = sess.run(
                prediction_op,
                feed_dict={ph_x: x, ph_x_len: lengths_x})

            prediction, att = process_beam_results(len(lengths_x), predict_out)
            y_str = [delimiter.join(vocab_target.convert_to_wordlist(prediction[sample_idx]))
                     for sample_idx in range(prediction.shape[0])]
            fw.write('\n'.join(y_str) + "\n")
            # random sample
            if random.random() < 0.2 and len(samples_src) < 5:
                for sample_idx in range(len(x_str)):
                    samples_src.append(x_str[sample_idx])
                    samples_trg.append(y_str[sample_idx])
                    if len(samples_src) >= 5:
                        break
            # output attention
            if output_attention:
                for idx in range(len(lengths_x)):
                    trans_list = vocab_target.convert_to_wordlist(prediction[idx, :], bpe=False)
                    attentions[cnt + idx] = {
                        "source": x_str[idx],
                        "translation": " ".join(trans_list),
                        "attention": att[:len(trans_list) + 1, idx, :lengths_x[idx]].tolist()}
            cnt += len(lengths_x)
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


def process_beam_results(batch_size, predict_out):
    """
    process beam results and return the best prediction sequence(word id sequence)
    :param batch_size: `int`, batch size
    :param predict_out: `dict`, from seq2seq_model output
    :return: predictions
    """
    predicted_ids = predict_out["predicted_ids"]  # [n_timesteps_trg, beam_size * batch_size]
    beam_ids = predict_out["beam_ids"]  # [n_timesteps_trg, beam_size * batch_size]
    sequence_lengths = predict_out["sequence_lengths"]  # [n_timesteps_trg, beam_size * batch_size]
    log_probs = predict_out["log_probs"]  # [n_timesteps_trg, beam_size * batch_size]

    gathered_pred_ids = numpy.zeros_like(beam_ids)
    for idx in range(beam_ids.shape[0]):
        gathered_pred_ids = gathered_pred_ids[:, beam_ids[idx]]
        gathered_pred_ids[idx, :] = predicted_ids[idx]
    gathered_pred_ids = gathered_pred_ids.transpose()

    log_probs = log_probs[-1] / numpy.array(sequence_lengths[-1])

    beam_size = log_probs.shape[0] // batch_size
    log_probs = log_probs.reshape([-1, beam_size])
    argmax_idx = numpy.argmax(log_probs, axis=1)  # [batch_size]
    argmax_idx += numpy.arange(batch_size) * beam_size

    if "attention_scores" in predict_out:
        # [n_timesteps_trg, beam_size*batch_size, n_timesteps_src]
        attention_scores = predict_out["attention_scores"]
        gathered_att = numpy.zeros_like(attention_scores)
        for idx in range(beam_ids.shape[0]):
            gathered_att = gathered_att[:, beam_ids[idx], :]
            gathered_att[idx, :, :] = attention_scores[idx]
        return gathered_pred_ids[argmax_idx, :], gathered_att[:, argmax_idx, :]
    return gathered_pred_ids[argmax_idx, :], None
