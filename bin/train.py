# -*- coding: utf-8 -*-
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
"""Entrance for training NMT models. """
import os
import tensorflow as tf
from tensorflow import gfile
from njunmt.nmt_experiment import TrainingExperiment
from njunmt.utils.configurable import update_configs_from_flags
from njunmt.utils.configurable import load_from_config_path
from njunmt.utils.configurable import define_tf_flags

# define arguments for train.py
# format: {arg_name: [type, default_val, helper]}
TRAIN_ARGS = {
    "config_paths": ["string", "", """Path to a yaml configuration files defining FLAG values.
                                   Multiple files can be separated by commas. Files are merged recursively.
                                   Setting a key in these files is equivalent to
                                   setting the FLAG value with the same name."""],
    "model_dir": ["string", "models", """The path to save models. """],
    "problem_name": ["string", "seq2seq", """The name for this run."""],
    "train": ["string", "", """A yaml-style string defining the training options."""],
    "data": ["string", "", """A yaml-style string defining the training data files,
                            evaluation data files, vocabulary files and bpe codes."""],
    "hooks": ["string", "", """A yaml-style string defining the training hooks (if implemented). """],
    "metrics": ["string", "", """A yaml-style string defining the evaluation metrics for training steps (if implemented),
                                e.g. BLEU, crossentropy loss."""],
    "model": ["string", "", """SequenceToSequence", "The model class name."""],
    "model_params": ["string", "", """A yaml-style string defining the model parameters."""],
    "optimizer_params": ["string", "", """A yaml-style string defining the parameters for optimizer."""]
}

FLAGS = define_tf_flags(TRAIN_ARGS)


def main(_argv):
    # load flags from config file
    model_configs = load_from_config_path(FLAGS.config_paths)
    # replace parameters in configs_file with tf FLAGS
    model_configs = update_configs_from_flags(model_configs, FLAGS, TRAIN_ARGS.keys())
    model_dir = model_configs["model_dir"]
    if not gfile.Exists(model_dir):
        gfile.MakeDirs(model_dir)

    if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        raise OSError("need CUDA_VISIBLE_DEVICES environment variable")
    tf.logging.info("CUDA_VISIBLE_DEVICES={}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    training_runner = TrainingExperiment(
        model_configs=model_configs)

    training_runner.run()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
