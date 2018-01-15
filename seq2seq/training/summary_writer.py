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

from tensorflow.python.framework import ops
from tensorflow.python.summary.writer.writer import FileWriter
from tensorflow.core.framework.summary_pb2 import Summary


class SummaryWriter(object):
    __instance = None

    def __new__(cls, logdir):
        assert logdir is not None and logdir != "", \
            "need model_dir to initialize SummaryWriter"
        if SummaryWriter.__instance is None:
            SummaryWriter.__instance = super(
                SummaryWriter, cls).__new__(cls)
            fw = FileWriter(logdir, graph=ops.get_default_graph())
            setattr(SummaryWriter.__instance, "_summary_writer", fw)
            setattr(SummaryWriter.__instance, "add_graph", fw.add_graph)
            setattr(SummaryWriter.__instance, "add_meta_graph", fw.add_meta_graph)
            setattr(SummaryWriter.__instance, "add_session_log", fw.add_session_log)
        return SummaryWriter.__instance

    def add_summary(self, summary_tag, summary_value, global_step):
        summary = Summary(value=[Summary.Value(
            tag=summary_tag, simple_value=summary_value)])
        self._summary_writer.add_summary(summary, global_step)

    @staticmethod
    def get_instance():
        assert SummaryWriter.__instance is not None, \
            "initialize SummaryWriter with model_dir first"
        return SummaryWriter.__instance

    @staticmethod
    def init_instance(model_dir):
        SummaryWriter(model_dir)
