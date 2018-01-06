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
"""Entrance for inference from a trained NMT model. """
import tensorflow as tf

from njunmt.utils.configurable import ModelConfigs
from njunmt.utils.configurable import update_infer_model_configs
from njunmt.utils.configurable import deep_merge_dict
from njunmt.utils.configurable import DEFAULT_INFER_CONFIGS
from njunmt.utils.configurable import maybe_load_yaml
from njunmt.utils.configurable import load_from_config_path
from njunmt.nmt_experiment import InferExperiment
from njunmt.ensemble_experiment import EnsembleExperiment

tf.flags.DEFINE_string("config_paths", "", """Path to a yaml configuration files defining FLAG
                       values. Multiple files can be separated by commas.
                       Files are merged recursively. Setting a key in these
                       files is equivalent to setting the FLAG value with
                       the same name.""")

tf.flags.DEFINE_string("infer", "", "inference options")
tf.flags.DEFINE_string("infer_data", "", "inference for data")
tf.flags.DEFINE_string("model_dir", "",
                       """model directory""")
tf.flags.DEFINE_string("weight_scheme", "average",
                       "weight scheme for ensemble, by default: average")
FLAGS = tf.flags.FLAGS


def main(_argv):
    model_configs = maybe_load_yaml(DEFAULT_INFER_CONFIGS)
    # load flags from config file
    model_configs = load_from_config_path(FLAGS.config_paths, model_configs)
    # replace parameters in configs_file with tf FLAGS
    model_configs = update_infer_model_configs(model_configs, FLAGS)

    model_dirs = FLAGS.model_dir.strip().split(",")
    if len(model_dirs) == 1:
        model_configs = deep_merge_dict(model_configs, ModelConfigs.load(model_dirs[0]))
        model_configs = update_infer_model_configs(model_configs, FLAGS)
        runner = InferExperiment(model_configs=model_configs)
    else:
        runner = EnsembleExperiment(model_configs=model_configs, model_dirs=model_dirs,
                                    weight_scheme=FLAGS.weight_scheme)
    runner.run()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
