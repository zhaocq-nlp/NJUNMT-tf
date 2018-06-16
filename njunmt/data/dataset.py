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

from njunmt.utils.misc import access_multiple_files


class Dataset_new(object):
    """ Class for training data and evaluation data """

    def __init__(self,
                 vocab_source,
                 vocab_target,
                 features_file,
                 labels_file=None):
        """ Initializes the data set.

        Args:
            vocab_source: A `Vocab` instance for source side.
            vocab_target: A `Vocab` instance for target side.
            features_file: A string.
            labels_file: A string.
        """
        self._vocab_source = vocab_source
        self._vocab_target = vocab_target
        self._features_file = features_file
        self._labels_file = labels_file

    @property
    def vocab_source(self):
        """ Returns the source vocab. """
        return self._vocab_source

    @property
    def vocab_target(self):
        """ Returns the target vocab. """
        return self._vocab_target

    @property
    def features_file(self):
        """ Returns the name of source file. """
        return self._features_file

    @property
    def labels_file(self):
        """ Returns the name of target file. """
        return self._labels_file


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
            self._eval_labels_file = [access_multiple_files(f) for f in eval_labels_file]
        elif isinstance(eval_labels_file, six.string_types):
            self._eval_labels_file = access_multiple_files(eval_labels_file)
        else:
            self._eval_labels_file = None

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
