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
""" Define utility functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import socket

import numpy
import tensorflow as tf
from tensorflow import gfile
from tensorflow.python.client import device_lib

from njunmt.utils.global_names import GlobalNames


def open_file(filename, encoding="utf-8", mode="r"):
    """ Opens file using codecs module.

    Args:
        filename: A string.
        encoding: A string specifies the encoding which is to be used for the
          file.
        mode: A string epecifies the opening mode.

    Returns: A file descriptor.
    """
    if mode == "r" and not gfile.Exists(filename):
        raise OSError("File: \"{}\" not exists.".format(filename))
    return codecs.open(filename, mode=mode, encoding=encoding)


def close_file(fp):
    """ Closes a file descriptor.

    Args:
        fp: A file descriptor.
    """
    if not fp.closed:
        fp.close()


def port_is_open(host):
    """ Checks whether the port is open.

    Args:
        host: A string has format "ip:port".

    Returns: True if the port is open, False otherwise.
    """
    ip, port = host.strip().split(":")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        # print '%d is open' % port
        return True
    except:
        # print '%d is down' % port
        return False


def create_ps_worker(ps_hosts, worker_hosts, task_index, ps):
    """ Creates tf ps and workers.

    Args:
        ps_hosts: A list of host strings with format "ip:port".
        worker_hosts: A list of worker strings with format "ip:port".
        task_index: The task index.
        ps: Whether it is a parameter server.

    Returns: A tuple `(server, clusters, num_workers, gpu_options)`.
    """
    ps_hosts = ps_hosts
    worker_hosts = worker_hosts
    num_workers = len(worker_hosts)

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    gpu_options = tf.GPUOptions(allocator_type='BFC', allow_growth=True)
    if ps:
        for host in ps_hosts:
            if port_is_open(host):
                raise ValueError("Error with ps_hosts: %s, the port %s is already occupied." \
                                 % (host, host.split(":")[1]))
        server_def = tf.train.ServerDef(cluster=cluster.as_cluster_def(),
                                        job_name="ps",
                                        task_index=task_index,
                                        default_session_config=tf.ConfigProto(gpu_options=gpu_options,
                                                                              device_count={"GPU": 0}),
                                        protocol="grpc")
    else:
        host = worker_hosts[task_index]
        if port_is_open(host):
            raise ValueError("Error with worker_hosts: %s, the port %s is already occupied." \
                             % (host, host.split(":")[1]))
        server_def = tf.train.ServerDef(cluster=cluster.as_cluster_def(),
                                        job_name="worker",
                                        task_index=task_index,
                                        default_session_config=tf.ConfigProto(gpu_options=gpu_options),
                                        protocol="grpc")
    server = tf.train.Server(server_def)
    return server, cluster, num_workers, gpu_options


def dump_model_analysis(model_dir):
    """ Dumps detailed model size.

    Args:
        model_dir: The directory name to save to.
    """
    # Dump to file on the chief worker
    filename = os.path.join(model_dir, GlobalNames.MODEL_ANALYSIS_FILENAME)
    profile_opt_builder = tf.profiler.ProfileOptionBuilder
    opts = profile_opt_builder.trainable_variables_parameter()
    opts["output"] = "file:outfile={}".format(filename)
    param_stats = tf.profiler.profile(tf.get_default_graph(), options=opts)
    # following APIs are deprecated
    # opts = tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS
    # opts['dump_to_file'] = os.path.abspath(filename)
    # tf.contrib.tfprof.model_analyzer.print_model_analysis(
    #     tf.get_default_graph(), tfprof_options=opts)
    # Print the model analysis
    with gfile.GFile(filename) as file:
        tf.logging.info(file.read())


def get_available_gpus():
    """Returns a list of available GPU devices names. """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]


def label_smoothing(labels, vocab_size, epsilon=0.1):
    """Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Args:
      labels: A 2d tensor with shape of [N, T].
      vocab_size: The size of vocabulary.
      epsilon: Smoothing rate.

    Returns: The smoothed labels.

    For example,
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
    outputs = label_smoothing(inputs)
    with tf.Session() as sess:
        print(sess.run([outputs]))
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    """
    confidence = 1. - epsilon
    low_confidence = epsilon / tf.to_float(vocab_size - 1)
    normalizing = -(confidence * tf.log(confidence)
                    + tf.to_float(vocab_size - 1) * low_confidence
                    * tf.log(low_confidence + 1e-20))
    soft_targets = tf.one_hot(
        indices=labels,
        depth=vocab_size,
        on_value=confidence,
        off_value=low_confidence)
    return soft_targets, normalizing


def padding_batch_data(seqs_x, padding_x):
    """ Creates batch data tensor.

    Args:
        seqs_x: A list of word sequence ids. Each word sequence is also
          a list.
        padding_x: The symbol id to be added to empty position.

    Returns: A tuple `(seqs, seq_lengths)`, where `seqs` is a 2-d
      numpy.ndarray with shape [len(seqs_x), max_seq_len] and
      `seq_lengths` is a 1-d numpy.ndarray with shape [len(seqs_x), ].

    """
    lengths_x = [len(s) for s in seqs_x]
    max_len_x = numpy.max(lengths_x)
    n_samples = len(seqs_x)
    x = numpy.full([n_samples, max_len_x], padding_x, numpy.int32)
    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
    return x, numpy.array(lengths_x, dtype=numpy.int32)


def add_dict_to_collection(collection_name, dict_):
    """ Adds a dictionary to a graph collection.

    Args:
        collection_name: The name of the collection to add the dictionary to.
        dict_: A dictionary of string keys to tensor values.
    """
    key_collection = collection_name + "_keys"
    value_collection = collection_name + "_values"
    for key, value in dict_.items():
        tf.add_to_collection(key_collection, key)
        tf.add_to_collection(value_collection, value)


def get_dict_from_collection(collection_name):
    """ Gets a dictionary from a graph collection.

    Args:
        collection_name: A collection name to read a dictionary from.

    Returns: A dictionary with string keys and tensor values
    """
    key_collection = collection_name + "_keys"
    value_collection = collection_name + "_values"
    keys = tf.get_collection(key_collection)
    values = tf.get_collection(value_collection)
    return dict(zip(keys, values))


def deprecated(obj):
    """This is a decorator which can be used to mark functions or classes
    as deprecated. It will result in a warning being emmitted
    when the function/class is used."""

    def new_obj(*args, **kwargs):
        tf.logging.info("Call to deprecated function/class %s." % obj.__name__)
        tf.logging.warn("Call to deprecated function/class %s." % obj.__name__)
        return obj(*args, **kwargs)

    return new_obj


def shuffle_data(from_binding, to_binding):
    """ Calls njunmt/tools/shuffle.py to shuffle data.

    Args:
        from_binding: The original data files with same number of lines.
        to_binding: The files to save to.
    """
    cmd = "python {script} {from_} {to_}".format(
        script="njunmt/tools/shuffle.py",
        from_=",".join(from_binding),
        to_=",".join(to_binding))
    os.system(cmd)


def get_labels_files(labels_file):
    """ Gets the list of labels file.

    Args:
        labels_file: A string, the prefix of the labels file.

    Returns: A list or None.
    """
    if labels_file is None:
        return None
    ret = []
    if gfile.Exists(labels_file):
        ret.append(labels_file)
    else:
        idx = 0
        while gfile.Exists(labels_file + str(idx)):
            ret.append(labels_file + str(idx))
            idx += 1
    return ret
