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


class Dataset(object):
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
