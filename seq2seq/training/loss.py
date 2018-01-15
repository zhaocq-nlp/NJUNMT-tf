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
import tensorflow as tf


def crossentropy_loss(logits, targets, sequence_length):
    """

    :param logits: [n_timesteps_trg, batch_size, vocab_size_trg]
    :param targets: [n_timesteps_trg, batch_size]
    :param sequence_length: [[batch_size], tf.int32
    :return:
    """
    # [n_timesteps_trg, batch_size]
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=targets)

    # [n_timesteps_trg, batch_size]
    loss_mask = tf.transpose(
        tf.sequence_mask(
            lengths=tf.to_int32(sequence_length),
            maxlen=tf.to_int32(tf.shape(targets)[0]),
            dtype=tf.float32), [1, 0])

    # mask loss
    losses = losses * loss_mask

    # average loss
    avg_length = tf.to_float(sequence_length)
    loss = tf.reduce_mean(
        tf.reduce_sum(losses, axis=0) / avg_length)
    return loss
