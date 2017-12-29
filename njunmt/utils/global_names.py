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
"""Predefined ModeKeys, file names, placeholder names and so on."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


ModeKeys = tf.contrib.learn.ModeKeys


class GlobalNames:
    """ Class to access predefined strings. """
    # for BLEU metric temp translation output directionary
    TMP_TRANS_DIRNAME = "translations"

    # for BLEU metric temp translation filename
    TMP_TRANS_FILENAME_PREFIX = "trans_"

    # for BLEU metric temp logging best bleu & checkpoint file
    TOP_BLEU_CKPTLOG_FILENAME = "topbleu_ckptnames.txt"

    # for BLEU metric temp reference filename (capable for bpe)
    TMP_REFERENCE_FILENAME = "reference"

    # for DevBLEUHelper checkpoint tgz filename prefix
    CKPT_TGZ_FILENAME_PREFIX = "checkpoint.iter"

    # for runner, model analysis filename
    MODEL_ANALYSIS_FILENAME = "model_analysis.txt"

    # for Saver ckpt filename
    MODEL_CKPT_FILENAME = "model-ckpt"

    # train options json filename
    MODEL_CONFIG_YAML_FILENAME = "model_configs.yml"

    # learning rate variable name
    LEARNING_RATE_VAR_NAME = "learning_rate"
    # for loss decay
    LR_ANNEAL_DIV_FACTOR_NAME = "lr_anneal_div_factor"

    # training loss display name
    TRAIN_LOSS_KEY_NAME = "training/loss"
    # collection name for key strs for tensors to be displayed
    DISPLAY_KEY_COLLECTION_NAME = "display_tensors_key"
    DISPLAY_VALUE_COLLECTION_NAME = "display_tensors_value"

    # placeholder prefix
    PH_PREFIX = "ph_"

    # default placeholders
    PH_FEATURE_IDS_NAME = PH_PREFIX + "feature_ids"
    PH_FEATURE_LENGTH_NAME = PH_PREFIX + "feature_length"
    PH_LABEL_IDS_NAME = PH_PREFIX + "label_ids"
    PH_LABEL_LENGTH_NAME = PH_PREFIX + "label_length"

    # verbose prefix for training hooks
    HOOK_VERBOSE_PREFIX = " ---hook order: "

    # ensemble model namescope prefix
    ENSEMBLE_VARNAME_PREFIX = "ensemble"
