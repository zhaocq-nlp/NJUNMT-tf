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
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
from tensorflow import gfile
import json
import time

import seq2seq
from seq2seq.utils.configurable import _deep_merge_dict
from seq2seq.utils.utils import optimistic_restore
from seq2seq.utils.global_names import GlobalNames
from seq2seq.data.vocab import Vocab
from seq2seq.inference.decode import infer
from seq2seq.data.data_iterator import TestTextIterator

tf.flags.DEFINE_string("model_dir", "",
                       """model directory""")

tf.flags.DEFINE_string("vocab_source", "", "source side vocabulary")
tf.flags.DEFINE_string("vocab_target", "", "target side vocabulary")
tf.flags.DEFINE_string("infer_source", "",
                       """file to be infered""")
tf.flags.DEFINE_integer("max_seq_len", 100,
                        """maximum sequence length""")
tf.flags.DEFINE_string("output", "",
                       """output file""")
tf.flags.DEFINE_string("delimiter", "",
                       """delimiter for output""")
tf.flags.DEFINE_integer("beam_size", 5, """beam size""")
tf.flags.DEFINE_integer("batch_size", 1, """batch size""")
tf.flags.DEFINE_boolean("use_bpe", False,
                        """target side use bpe""")
tf.flags.DEFINE_boolean("output_attention", False,
                        """whether to output attention""")

FLAGS = tf.flags.FLAGS


def _default_model_configs():
    return {
        "model": "seq2seq.models.base_seq2seq.BaseSeq2Seq",
        "model_params": {}
    }


def main(_argv):
    model_configs_file = os.path.join(FLAGS.model_dir, GlobalNames.MODEL_CONFIG_JSON_FILENAME)
    if not gfile.Exists(model_configs_file):
        raise OSError("Fail to open model configs file: %s. File does not exist..." % model_configs_file)
    with gfile.GFile(model_configs_file) as file:
        model_configs = json.load(file)
    model_configs = _deep_merge_dict(_default_model_configs(), model_configs)

    if FLAGS.vocab_source == "" or not os.path.exists(FLAGS.vocab_source):
        tf.logging.info('Source vocab: %s not found', FLAGS.vocab_source)
        return
    if FLAGS.vocab_target == "" or not os.path.exists(FLAGS.vocab_target):
        tf.logging.info('Target vocab: %s not found', FLAGS.vocab_target)
        return

    model_configs['model_params']['inference.beam_size'] = FLAGS.beam_size
    model_configs['model_params']['inference.max_seq_len'] = FLAGS.max_seq_len
    vocab_source = Vocab(filename=FLAGS.vocab_source)
    vocab_target = Vocab(filename=FLAGS.vocab_target, use_bpe=FLAGS.use_bpe)

    seq2seqmodel = eval(model_configs["model"])(
        params=model_configs["model_params"],
        mode=tf.contrib.learn.ModeKeys.INFER,
        vocab_source=vocab_source,
        vocab_target=vocab_target)

    source_ids = tf.placeholder(tf.int32, shape=(None, None), name='source_ids')
    source_seq_length = tf.placeholder(tf.int32, shape=(None,), name='source_seq_length')
    input_fields = {GlobalNames.PH_SOURCE_SEQIDS_NAME: source_ids,
                    GlobalNames.PH_SOURCE_SEQLENGTH_NAME: source_seq_length}
    seq2seq_info = seq2seqmodel.build(input_fields)

    predict_op = seq2seq_info.predictions

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    test_data = TestTextIterator(FLAGS.infer_source,
                                 vocab_source,
                                 FLAGS.batch_size)

    # reload
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    if checkpoint_path:
        tf.logging.info("reloading models...")
        optimistic_restore(sess, checkpoint_path)
    else:
        raise OSError("File NOT Found. Fail to find checkpoint file from: %s" % FLAGS.model_dir)

    tf.logging.info("start infering...")
    start_time = time.time()

    infer(sess=sess,
          input_fields=input_fields,
          prediction_op=predict_op,
          testdata_iterator=test_data,
          output=FLAGS.output,
          vocab_target=vocab_target,
          delimiter=FLAGS.delimiter,
          output_attention=FLAGS.output_attention)

    tf.logging.info("FINISHED......")
    tf.logging.info("Elapsed Time: %s" % str(time.time() - start_time))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
