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
from njunmt.utils.configurable import maybe_load_yaml, DEFAULT_TRAIN_CONFIGS
from njunmt.utils.configurable import update_train_model_configs
from njunmt.utils.configurable import load_from_config_path

tf.flags.DEFINE_string("config_paths", "",
                       """Path to a yaml configuration files defining FLAG
                       values. Multiple files can be separated by commas.
                       Files are merged recursively. Setting a key in these
                       files is equivalent to setting the FLAG value with
                       the same name.""")

tf.flags.DEFINE_string("model_dir", "", """model directory""")
tf.flags.DEFINE_string("problem_name", None, """problem name""")
tf.flags.DEFINE_string("train", "", """training options""")
tf.flags.DEFINE_string("data", "", """training and evaluation data files, vocabulary files, bpe codes""")
tf.flags.DEFINE_string("hooks", "", """training hooks""")
tf.flags.DEFINE_string("metrics", "", """evaluation metrics""")
tf.flags.DEFINE_string("model", "", """model class name""")
tf.flags.DEFINE_string("model_params", "", """model parameters""")
tf.flags.DEFINE_string("optimizer_params", "", """optimizer parameters""")

tf.flags.DEFINE_integer("task_index", -1, """task index for multi-GPU""")  # task index for multi-gpu

FLAGS = tf.flags.FLAGS


def main(_argv):
    model_configs = maybe_load_yaml(DEFAULT_TRAIN_CONFIGS)
    # load flags from config file
    model_configs = load_from_config_path(FLAGS.config_paths, model_configs)
    # replace parameters in configs_file with tf FLAGS
    model_configs = update_train_model_configs(model_configs, FLAGS)
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
