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

import seq2seq
from seq2seq.data.vocab import Vocab
from seq2seq.data.data_iterator import TrainTextIterator
from seq2seq.utils.configurable import _maybe_load_json, _deep_merge_dict, ModelConfigs
from seq2seq.training.training_hooks import build_training_hooks
from seq2seq.utils.global_names import GlobalNames
from seq2seq.utils.utils import create_train_op
from collections import namedtuple

tf.flags.DEFINE_string("config_paths", "",
                       """Path to a json configuration files defining FLAG
                       values. Multiple files can be separated by commas.
                       Files are merged recursively. Setting a key in these
                       files is equivalent to setting the FLAG value with
                       the same name.""")

tf.flags.DEFINE_string("train_source", "", "source training data")
tf.flags.DEFINE_string("train_target", "", "target training data")

tf.flags.DEFINE_string("dev_source", "", "source development data")
tf.flags.DEFINE_string("dev_target", "", "target development data")

tf.flags.DEFINE_string("training_params", "",
                       """parameters for training""")

tf.flags.DEFINE_string("training_options", "",
                       """options for training""")

tf.flags.DEFINE_string("output_dir", "./models",
                       """The directory to write model checkpoints and summaries
                       to. If None, a local temporary directory is created.""")
tf.flags.DEFINE_boolean("use_bpe", False,
                        """target side use bpe""")

FLAGS = tf.flags.FLAGS


def _default_training_params():
    return {
        "vocab_source": "",
        "vocab_target": "",
        "batch_size": 80,
        "source_max_seq_len": 50,
        "target_max_seq_len": 50,
        "shuffle_every_epoch": None,
        "shuffle_before_train": None
    }


def _default_training_options():
    return {
        "train_steps": 1000000,
        "train_epochs": 30,
        "display_steps": 100,
        "save_checkpoints_every_n_steps": 1000
    }


def _default_model_configs():
    return {
        "model": "seq2seq.models.base_seq2seq.BaseSeq2Seq",
        "model_params": {}
    }


training_configs_type = namedtuple(
    typename="training_configs",
    field_names=["training_options",
                 "model_configs",
                 "model_dir",
                 "train_source",
                 "train_target",
                 "dev_source",
                 "dev_target",
                 "vocab_source",
                 "vocab_target",
                 "ops",
                 "input_fields"])


def main(_argv):
    # check data file
    if FLAGS.train_source == "" or not gfile.Exists(FLAGS.train_source):
        raise ValueError("OPTERROR (train_source): file does not exist: {}".format(FLAGS.train_source))
    tf.logging.info("train source file: " + FLAGS.train_source)
    if FLAGS.train_target == "" or not gfile.Exists(FLAGS.train_target):
        raise ValueError("OPTERROR (train_target): file does not exist: {}".format(FLAGS.train_target))
    tf.logging.info("train target file: " + FLAGS.train_target)
    if FLAGS.dev_source == "" or not gfile.Exists(FLAGS.dev_source):
        raise ValueError("OPTERROR (dev_source): file does not exist: {}".format(FLAGS.dev_source))
    tf.logging.info("dev source file: " + FLAGS.dev_source)
    if FLAGS.dev_target == "" or (not gfile.Exists(FLAGS.dev_target) and not gfile.Exists(FLAGS.dev_target + "0")):
        raise ValueError("OPTERROR (dev_target): file does not exist: {}".format(FLAGS.dev_target))
    tf.logging.info("dev target file: " + FLAGS.dev_target)

    # load flags from config file
    model_configs = {}
    if FLAGS.config_paths:
        for config_path in FLAGS.config_paths.split(","):
            config_path = config_path.strip()
            if not config_path:
                continue
            if not os.path.exists(config_path):
                raise OSError("config file does not exists: %s", config_path)
            config_path = os.path.abspath(config_path)
            tf.logging.info("loading configurations from %s", config_path)
            with gfile.GFile(config_path.strip()) as config_file:
                config_flags = json.load(config_file)
                model_configs = _deep_merge_dict(model_configs, config_flags)
    model_configs = _deep_merge_dict(_default_model_configs(), model_configs)

    if not gfile.Exists(FLAGS.output_dir):
        gfile.MakeDirs(FLAGS.output_dir)
    # dump model configs
    ModelConfigs.dump(model_configs, FLAGS.output_dir)

    if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        raise OSError("need CUDA_VISIBLE_DEVICES environment variable")

    # load training parameters
    FLAGS.training_params = _deep_merge_dict(_default_training_params(),
                                             _maybe_load_json(FLAGS.training_params))
    tf.logging.info("Training parameters: ")
    for k, v in FLAGS.training_params.items():
        tf.logging.info("    %s: %s" % (k, v))

    # load training options
    FLAGS.training_options = _deep_merge_dict(_default_training_options(),
                                              _maybe_load_json(FLAGS.training_options))
    tf.logging.info("Training options:")
    for k, v in FLAGS.training_options.items():
        tf.logging.info("    %s: %s" % (k, v))

    # collect all training configs
    training_configs = training_configs_type(
        training_options=FLAGS.training_options,
        model_configs=model_configs,
        model_dir=FLAGS.output_dir,
        train_source=FLAGS.train_source,
        train_target=FLAGS.train_target,
        dev_source=FLAGS.dev_source,
        dev_target=FLAGS.dev_target,
        vocab_source=Vocab(filename=FLAGS.training_params["vocab_source"]),
        vocab_target=Vocab(filename=FLAGS.training_params["vocab_target"], use_bpe=FLAGS.use_bpe),
        ops=dict(),  # operations
        input_fields=dict())

    # Create model template function
    tf.logging.info("Create model: BaseSeq2Seq for TRAIN")
    seq2seq_model = eval(model_configs["model"])(
        params=model_configs["model_params"],
        mode=tf.contrib.learn.ModeKeys.TRAIN,
        vocab_source=training_configs.vocab_source,
        vocab_target=training_configs.vocab_target)
    # build placeholders
    source_ids = tf.placeholder(tf.int32, shape=(None, None), name='source_ids')
    source_seq_length = tf.placeholder(tf.int32, shape=(None,), name='source_seq_length')
    target_ids = tf.placeholder(tf.int32, shape=(None, None), name='target_ids')
    target_seq_length = tf.placeholder(tf.int32, shape=(None,), name='target_seq_length')
    input_fields = {GlobalNames.PH_SOURCE_SEQIDS_NAME: source_ids,
                    GlobalNames.PH_SOURCE_SEQLENGTH_NAME: source_seq_length,
                    GlobalNames.PH_TARGET_SEQIDS_NAME: target_ids,
                    GlobalNames.PH_TARGET_SEQLENGTH_NAME: target_seq_length}
    # build models
    seq2seq_info = seq2seq_model.build(input_fields)
    loss = seq2seq_info.loss

    train_op, global_step_tensor = create_train_op(model_configs["optimizer_params"], loss)

    # auxiliary inputs for training hooks
    training_configs.ops["loss_op"] = loss
    training_configs.ops["global_step_tensor"] = global_step_tensor
    training_configs.input_fields.update(input_fields)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.train.MonitoredSession(
        session_creator=None,
        hooks=build_training_hooks(training_configs))

    max_train_epochs = FLAGS.training_options['train_epochs']

    train_data = TrainTextIterator(FLAGS.train_source,
                                   FLAGS.train_target,
                                   training_configs.vocab_source,
                                   training_configs.vocab_target,
                                   maxlen_src=FLAGS.training_params["source_max_seq_len"],
                                   maxlen_trg=FLAGS.training_params["target_max_seq_len"],
                                   batch_size=FLAGS.training_params['batch_size'],
                                   shuffle_before_train=FLAGS.training_params["shuffle_before_train"],
                                   shuffle_every_epoch=FLAGS.training_params["shuffle_every_epoch"])

    for eidx in range(max_train_epochs):
        if sess.should_stop():
            break
        tf.logging.info("STARTUP Epoch %d" % eidx)

        for (x, lengths_x), (y, lengths_y) in train_data:
            if sess.should_stop():
                tf.logging.info("Training maximum steps...")
                break
            sess.run([train_op],
                     feed_dict={source_ids: x, source_seq_length: lengths_x,
                                target_ids: y, target_seq_length: lengths_y})


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
