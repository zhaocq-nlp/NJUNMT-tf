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

import os
import numpy
import tensorflow as tf
from tensorflow import gfile

from seq2seq.utils.global_names import GlobalNames


def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            # print saved_var_name, type(curr_var)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


def dump_model_analysis(model_dir):
    # Dump to file on the chief worker
    filename = os.path.join(model_dir, GlobalNames.MODEL_ANALYSIS_FILENAME)
    opts = tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS
    opts['dump_to_file'] = os.path.abspath(filename)
    tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(), tfprof_options=opts)
    # Print the model analysis
    with gfile.GFile(filename) as file:
        tf.logging.info(file.read())


def pedding_batch_data(seqs_x, padding_x):
    """ pedding batch data

    :param seqs_x: `list`, sequences of sequence of source id
    :param padding_x: `int`, pedding id
    :return:
    """
    lengths_x = [len(s) for s in seqs_x]

    max_len_x = numpy.max(lengths_x)
    n_samples = len(seqs_x)

    x = numpy.full([n_samples, max_len_x], padding_x, numpy.int32)

    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
    return x, numpy.array(lengths_x, dtype=numpy.int32)


def create_train_op(params, loss, colocate_gradients_with_ops=False):
    """
    create training operator with learning annealing
    :param loss:
    :param colocate_gradients_with_ops:
    :return:
    """
    learning_rate = tf.get_variable(GlobalNames.LEARNING_RATE_VAR_NAME,
                                                shape=(), dtype=tf.float32,
                                                initializer=tf.constant_initializer(
                                                    value=params["optimizer.learning_rate"],
                                                    dtype=tf.float32),
                                                trainable=False)
    name = params["optimizer.name"]
    tf.logging.info("use %s optimizer with initial learning rate=%f"
                 % (name, params["optimizer.learning_rate"]))
    optimizer = tf.contrib.layers.OPTIMIZER_CLS_NAMES[name](
        learning_rate=learning_rate,
        **params["optimizer.params"])

    """Creates the training operation"""
    def _clip_gradients(grads_and_vars):
        """Clips gradients by global norm."""
        gradients, variables = zip(*grads_and_vars)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, params["optimizer.clip_gradients"])
        return list(zip(clipped_gradients, variables))

    global_step_tensor = tf.get_variable(name=GlobalNames.GLOBAL_STEP_VAR_NAME,
                                                     dtype=tf.int32, shape=(),
                                                     initializer=tf.constant_initializer(
                                                         value=0, dtype=tf.int32),
                                                     trainable=False)
    # global_step_tensor = tf.Variable(0, name="global_step", dtype=tf.int32)
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step_tensor,
        learning_rate=params["optimizer.learning_rate"],
        learning_rate_decay_fn=None,
        clip_gradients=_clip_gradients,
        optimizer=optimizer,
        summaries=["learning_rate", "loss"],
        colocate_gradients_with_ops=colocate_gradients_with_ops)

    return train_op, global_step_tensor
