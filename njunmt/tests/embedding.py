# Copyright 2017 ZhaoChengqi, zhaocq.nlp@gmail.com, Natural Language Processing Group, Nanjing University (2015-2018).
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

import math

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import embedding_ops

from njunmt.utils.misc import deprecated


@deprecated
class WordEmbedder(object):
    def __init__(self,
                 vocab_size,
                 dimension,
                 init_scale=0.04,
                 maximum_position=300,
                 timing=None,
                 name=None):
        """ initialize embedding table

        :param vocab_size: vocabulary size
        :param dimension: dimension of embedding
        :param init_scale: init scale
        :param timing: position embedding type, "sinusoids" or "emb"
        :param name: scope name of embedding table
        """
        self._vocab_size = vocab_size
        self._dimension = dimension
        self._init_scale = init_scale
        self._maximum_position = maximum_position
        self._timing = timing
        self._name = name
        assert timing in ["sinusoids", "emb", None], \
            "timing should be one of \"sinusoids\", \"emb\" or None"
        self._build()

    def _build(self):
        """ build embedding table and
        build position embedding table if timing=="emb"

        :return:
        """
        self._embeddings = variable_scope.get_variable(
            name=(self._name or "embedding_table"),
            shape=[self._vocab_size, self._dimension],
            initializer=init_ops.random_uniform_initializer(
                -self._init_scale, self._init_scale))
        if self._timing == "emb":
            self._position_embedding = variable_scope.get_variable(
                name=(self._name or "embedding_table") + "_posi",
                shape=[self._maximum_position, self._dimension],
                initializer=init_ops.random_uniform_initializer(
                    -self._init_scale, self._init_scale))

    def get_shape(self):
        return self._embeddings.get_shape()

    def embed_words(self, words, time=0):
        """ embed the word

        :param words: 1/2-dim tensor, the first dimension indicates batch_size
        :param time: indicating the position
        :return: embeddings: [batch_size, length, dim_emb]
        """
        emb = embedding_ops.embedding_lookup(self._embeddings, words)
        return self._add_timing_signal(emb, time)

    @staticmethod
    def _add_sinusoids_signal(x, time, min_timescale=1.0, max_timescale=1.0e4):
        """Adds a bunch of sinusoids of different frequencies to a Tensor.

        Each channel of the input Tensor is incremented by a sinusoid of a different
        frequency and phase.

        This allows attention to learn to use absolute and relative positions.
        Timing signals should be added to some precursors of both the query and the
        memory inputs to attention.

        The use of relative position is possible because sin(x+y) and cos(x+y) can be
        experessed in terms of y, sin(x) and cos(x).

        In particular, we use a geometric sequence of timescales starting with
        min_timescale and ending with max_timescale.  The number of different
        timescales is equal to channels / 2. For each timescale, we
        generate the two sinusoidal signals sin(timestep/timescale) and
        cos(timestep/timescale).  All of these sinusoids are concatenated in
        the channels dimension.

        Args:
          x: a Tensor with shape [batch, length, channels]
          min_timescale: a float
          max_timescale: a float

        Returns:
          a Tensor the same shape as x.
        """
        channels = x.get_shape().as_list()[-1]
        if x.get_shape().ndims == 3:  # [batch_size, timesteps, dim]
            length = array_ops.shape(x)[1]
            position = math_ops.to_float(math_ops.range(length))
        elif x.get_shape().ndims == 2:  # [batch_size, dim]
            length = 1
            position = math_ops.to_float(math_ops.range(time, time + 1))
        else:
            raise ValueError("need a Tensor with rank 2 or 3")
        num_timescales = channels // 2
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (math_ops.to_float(num_timescales) - 1))
        inv_timescales = min_timescale * math_ops.exp(
            math_ops.to_float(math_ops.range(num_timescales)) * -log_timescale_increment)
        scaled_time = array_ops.expand_dims(position, 1) * array_ops.expand_dims(inv_timescales, 0)
        signal = array_ops.concat([math_ops.sin(scaled_time), math_ops.cos(scaled_time)], axis=1)
        signal = array_ops.pad(signal, [[0, 0], [0, math_ops.mod(channels, 2)]])
        if x.get_shape().ndims == 3:
            signal = array_ops.reshape(signal, [1, length, channels])
        else:
            signal = array_ops.reshape(signal, [1, channels])
        return x + signal

    def _add_emb_signal(self, x, time):
        """ add position embedding

        :param x:
        :param time:
        :return:
        """
        x_ndims = x.get_shape().ndims
        if x_ndims == 2:
            position = ops.convert_to_tensor(time, dtype=dtypes.int32)
        elif x_ndims == 3:
            position = math_ops.range(array_ops.shape(x)[1])
        else:
            raise ValueError("need a Tensor with rank 2 or 3")
        position_emb = embedding_ops.embedding_lookup(
            self._position_embedding, position)
        return x + array_ops.expand_dims(position_emb, 0)

    def _add_timing_signal(self, x, time=0):
        if self._timing is None:
            return x
        if self._timing == "sinusoids":
            return WordEmbedder._add_sinusoids_signal(x, time)
        if self._timing == "emb":
            return self._add_emb_signal(x, time)
