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

from abc import abstractmethod
from seq2seq.utils.configurable import Configurable


class Encoder(Configurable):
    def __init__(self, params, mode):
        """
        constructor for Encoder
        :param params: parameters
        :param mode: tf.contrib.learn.ModeKeys.TRAIN/EVAL/INFER
        :param scope: scope name
        """
        super(Encoder, self).__init__(params, mode)

    @staticmethod
    def default_params():
        raise NotImplementedError

    @abstractmethod
    def encode(self, *args, **kwargs):
        """
        Encodes the inputs
        :param args:
            inputs: [batch_size, timesteps, dim_input_emb]
            sequence_length: [batch_size,]
            There may be multiple inputs.
        :param kwargs:
        :return:
            EncoderOutput
        """
        raise NotImplementedError