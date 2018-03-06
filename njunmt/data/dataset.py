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
"""A class for training data and evaluation data"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from njunmt.utils.constants import Constants
from njunmt.utils.misc import get_labels_files


class Dataset(object):
    """ Class for training data and evaluation data """

    def __init__(self,
                 vocab_source,
                 vocab_target,
                 train_features_file=None,
                 train_labels_file=None,
                 eval_features_file=None,
                 eval_labels_file=None):
        """ Initializes the data set name for one training process.

        Args:
            vocab_source: A `Vocab` instance for source side feature map.
            vocab_target: A `Vocab` instance for target side feature map.
            train_features_file: The file name of training source file.
            train_labels_file: The file name of training target file.
            eval_features_file: The file name of evaluation source file.
            eval_labels_file: The file name of evaluation target file.
        """
        self._vocab_source = vocab_source
        self._vocab_target = vocab_target
        self._train_features_file = train_features_file
        self._train_labels_file = train_labels_file
        self._eval_features_file = eval_features_file
        if isinstance(eval_labels_file, list):
            self._eval_labels_file = [get_labels_files(f) for f in eval_labels_file]
        elif isinstance(eval_labels_file, six.string_types):
            self._eval_labels_file = get_labels_files(eval_labels_file)
        else:
            self._eval_labels_file = None
        self._input_fields = Dataset._make_input_fields()

    @staticmethod
    def _make_input_fields():
        """ Creates tf placeholders and add input data status
              to tf collections for displaying.

        Returns: A dictionary of placeholders.

        """
        feature_ids = array_ops.placeholder(dtypes.int32, shape=(None, None), name=Constants.FEATURE_IDS_NAME)
        feature_length = array_ops.placeholder(dtypes.int32, shape=(None,), name=Constants.FEATURE_LENGTH_NAME)
        label_ids = array_ops.placeholder(dtypes.int32, shape=(None, None), name=Constants.LABEL_IDS_NAME)
        label_length = array_ops.placeholder(dtypes.int32, shape=(None,), name=Constants.LABEL_LENGTH_NAME)

        feature_nonpadding_tokens_num = math_ops.reduce_sum(feature_length)
        feature_shape = array_ops.shape(feature_ids)
        feature_total_tokens_num = feature_shape[0] * feature_shape[1]
        label_nonpadding_tokens_num = math_ops.reduce_sum(label_length)
        label_shape = array_ops.shape(label_ids)
        label_total_tokens_num = label_shape[0] * label_shape[1]
        ops.add_to_collection(Constants.DISPLAY_KEY_COLLECTION_NAME, "input_stats/feature_nonpadding_tokens_num")
        ops.add_to_collection(Constants.DISPLAY_VALUE_COLLECTION_NAME, feature_nonpadding_tokens_num)
        ops.add_to_collection(Constants.DISPLAY_KEY_COLLECTION_NAME, "input_stats/feature_nonpadding_ratio")
        ops.add_to_collection(Constants.DISPLAY_VALUE_COLLECTION_NAME,
                              math_ops.to_float(feature_nonpadding_tokens_num)
                              / math_ops.to_float(feature_total_tokens_num))
        ops.add_to_collection(Constants.DISPLAY_KEY_COLLECTION_NAME, "input_stats/label_nonpadding_tokens_num")
        ops.add_to_collection(Constants.DISPLAY_VALUE_COLLECTION_NAME, label_nonpadding_tokens_num)
        ops.add_to_collection(Constants.DISPLAY_KEY_COLLECTION_NAME, "input_stats/label_nonpadding_ratio")
        ops.add_to_collection(Constants.DISPLAY_VALUE_COLLECTION_NAME,
                              math_ops.to_float(label_nonpadding_tokens_num)
                              / math_ops.to_float(label_total_tokens_num))
        return {Constants.FEATURE_IDS_NAME: feature_ids,
                Constants.FEATURE_LENGTH_NAME: feature_length,
                Constants.LABEL_IDS_NAME: label_ids,
                Constants.LABEL_LENGTH_NAME: label_length}

    @property
    def input_fields(self):
        """ Returns the dictionary of placeholders. """
        if self._input_fields is None:
            self._input_fields = Dataset._make_input_fields()
        return self._input_fields

    @property
    def vocab_source(self):
        """ Returns source side feature map (a `Vocab` object). """
        return self._vocab_source

    @property
    def vocab_target(self):
        """ Returns target side feature map (a `Vocab` object). """
        return self._vocab_target

    @property
    def train_features_file(self):
        """ Returns the file name of training source file. """
        return self._train_features_file

    @property
    def train_labels_file(self):
        """ Returns the file name of training target file. """
        return self._train_labels_file

    @property
    def eval_features_file(self):
        """ Returns the file name of evaluation source file. """
        return self._eval_features_file

    @property
    def eval_labels_file(self):
        """ Returns the file name of evaluation target file. """
        return self._eval_labels_file
