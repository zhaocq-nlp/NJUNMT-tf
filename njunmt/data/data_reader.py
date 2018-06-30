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
""" Class for reading in data lines. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import numpy

import tensorflow as tf

from njunmt.utils.misc import open_file, close_file
from njunmt.utils.misc import access_multiple_files


class LineReader(object):
    """ Class for reading in lines. """

    def __init__(self,
                 data,
                 maximum_length=None,
                 preprocessing_fn=None):
        """ Initializes the parameters for LineReader.

        Args:
            data: A string indicating the name of data file or a list of data list.
            maximum_length: An integer, the maximum length of one line (after
              preprocessed if `preprocessing_fn` is provided).
            preprocessing_fn: A callable function.
        """
        self._maximum_length = maximum_length
        self._preprocessing_fn = preprocessing_fn
        self._data_index = 0
        if isinstance(data, six.string_types):
            self._filename = access_multiple_files(data)[0]
            self._data = open_file(self._filename, encoding="utf-8", mode="r")
        elif isinstance(data, list):
            self._filename = None
            self._data = numpy.array(data)
        else:
            raise ValueError("Unrecognized type of `data`: {}, "
                             "which should be string or list".format(type(data)))

    def close(self):
        """ Closes this reader.  """
        self._data_index = 0
        if self._filename is not None:
            close_file(self._data)

    def reset(self, do_shuffle=False, shuffle_to_file=None, argsort_index=None):
        """ Resets this reader and shuffle (if needed).

        Args:
            do_shuffle: Whether to shuffle data.
            shuffle_to_file: A string.
            argsort_index: A list of integers

        Returns: The `argsort_index` if do shuffling.
        """
        # TODO
        self._data_index = 0
        if self._filename is not None:
            self._data.seek(0)
        if do_shuffle:
            if self._filename is None: # list of data
                _ = shuffle_to_file
                if not argsort_index:
                    argsort_index = numpy.arange(len(self._data))
                    numpy.random.shuffle(argsort_index)
                self._data = self._data[argsort_index]  # do shuffle
            else: # from file
                assert shuffle_to_file, (
                    "`shuffle_to_file` must be provided.")
                tf.logging.info("shuffling data:\t{} ==> {}".format(
                    self._filename, shuffle_to_file))
                data_list = self._data.readlines()
                close_file(self._data)
                if argsort_index is None:
                    argsort_index = numpy.arange(len(data_list))
                    numpy.random.shuffle(argsort_index)
                with open_file(shuffle_to_file,"utf-8", "w") as fw:
                    for idx in argsort_index:
                        fw.write(data_list[idx].strip() + "\n")
                del data_list[:]
                self._data = open_file(shuffle_to_file, "utf-8", "r")
        return argsort_index

    def next(self):
        """ Returns the next line (after preprocessing and filtering).
            Note that if reading from file, the file must contain no empty lines.

        Returns: "" if it hits the end of this data;
                None if this sample does not meet the requirement of `maximum_length`;
                A list of tokens, otherwise.
        """
        if self._filename is None:  # from a list of data
            line = "" if self._data_index >= len(self._data) \
                else self._data[self._data_index]
            self._data_index += 1
        else:  # from file
            line = self._data.readline()

        if line == "":
            return ""
        tokens = line.strip().split()
        if self._preprocessing_fn:
            tokens = self._preprocessing_fn(tokens)
        if self._maximum_length and len(tokens) > self._maximum_length:
            return None
        return tokens
