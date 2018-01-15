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


class GlobalNames:
    # for DevBLEUHelper temp translation directory
    TMP_TRANS_DIRNAME_PREFIX = "temp_trans_"

    # for DevBLEUHelper temp logging best bleu & checkpoint file
    TMP_BLEU_LOG_FILENAME = "tmp_log_bleu_file.txt"

    # for DevBLEUHelper temp translation filename
    TMP_TRANS_FILENAME_PREFIX = "trans_"

    # for DevBLEUHelper temp reference filename (capable for bpe)
    TMP_REFERENCE_FILENAME = "reference"

    # for DevBLEUHelper checkpoint tgz filename prefix
    CKPT_TGZ_FILENAME_PREFIX = "checkpoint.iter"

    # for runner, model analysis filename
    MODEL_ANALYSIS_FILENAME = "model_analysis.txt"

    # for Saver ckpt filename
    MODEL_CKPT_FILENAME = "model-ckpt"

    # train options json filename
    MODEL_CONFIG_JSON_FILENAME = "model_configs.json"

    # learning rate variable name
    LEARNING_RATE_VAR_NAME = "learning_rate"

    # global step var name
    GLOBAL_STEP_VAR_NAME = ops.GraphKeys.GLOBAL_STEP

    # placeholder prefix
    PH_PREFIX = "ph_"

    # default placeholders
    PH_SOURCE_SEQIDS_NAME = PH_PREFIX + "source_seqids"
    PH_SOURCE_SEQLENGTH_NAME = PH_PREFIX + "source_seqlength"
    PH_TARGET_SEQIDS_NAME = PH_PREFIX + "target_seqids"
    PH_TARGET_SEQLENGTH_NAME = PH_PREFIX + "target_seqlength"

    # verbose prefix for training hooks
    HOOK_VERBOSE_PREFIX = " ---hook order: "