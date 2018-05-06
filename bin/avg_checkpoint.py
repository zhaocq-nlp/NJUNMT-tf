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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoints", "",
                    "Comma-separated list of checkpoints to average.")
flags.DEFINE_string("output_path", "./averaged_ckpt",
                    "Path to output the averaged checkpoint to.")


def checkpoint_exists(path):
    return (tf.gfile.Exists(path) or tf.gfile.Exists(path + ".meta") or
            tf.gfile.Exists(path + ".index"))


def checkpoint_list_checking(path_list):
    if path_list:
        new_path_list = []
        for path in path_list:
            if checkpoint_exists(path):
                new_path_list.append(path)
        return new_path_list
    return []


def main(_):
    assert FLAGS.checkpoints

    model_config_yml_path = None

    # Get the checkpoints list from flags and run some basic checks.
    checkpoints = [c.strip() for c in FLAGS.checkpoints.split(",")]
    checkpoints = [c for c in checkpoints if c]
    if not checkpoints:
        raise ValueError("No checkpoints provided for averaging.")

    for c in checkpoints:
        if model_config_yml_path:
            break
        if tf.gfile.Exists(os.path.join(c, "model_configs.yml")):
            model_config_yml_path = os.path.join(c, "model_configs.yml")

    checkpoint_states = [tf.train.get_checkpoint_state(c) for c in checkpoints]

    checkpoints = sum([checkpoint_list_checking(s.all_model_checkpoint_paths) for s in checkpoint_states], [])
    if len(checkpoints) < 2:
        raise ValueError("Need more than 1 checkpoint to average")

    # Read variables from all checkpoints and average them.
    var_values = {}
    var_cnts = {}
    var_name_shape_list = tf.contrib.framework.list_variables(checkpoints[0])
    for ckpt in checkpoints:
        tf.logging.info("loading from {}".format(ckpt))
        for var_name, _ in var_name_shape_list:
            if var_name.startswith("OptimizeLoss"):
                continue
            if tf.GraphKeys.GLOBAL_STEP in var_name or "learning_rate" in var_name or "lr" in var_name:
                tf.logging.info("\tignore variable: {}".format(var_name))
                continue
            var = tf.contrib.framework.load_variable(ckpt, var_name)
            if var_name in var_values:
                var_cnts[var_name] += 1
                var_values[var_name] += var
            else:
                var_cnts[var_name] = 1
                var_values[var_name] = var
    # do average
    div_factor = float(len(checkpoints))
    for var_name in var_values.keys():
        assert var_cnts[var_name] == len(checkpoints)
        var_values[var_name] /= div_factor
        tf.get_variable(name=var_name, shape=var_values[var_name].shape,
                        dtype=tf.float32, initializer=tf.constant_initializer(var_values[var_name]))

    saver = tf.train.Saver(tf.all_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, FLAGS.output_path + "/model-ckpt", global_step=0)

    os.system("cp {} {}".format(model_config_yml_path, FLAGS.output_path))
    tf.logging.info("Averaged checkpoints saved in %s", FLAGS.output_path)
    with open(os.path.join(FLAGS.output_path, "avg_ckpt_list"), "w") as fw:
        fw.write("\n".join(checkpoints))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
