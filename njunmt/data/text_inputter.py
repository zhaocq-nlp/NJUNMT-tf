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
""" Classes for reading in data. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

import numpy
import six
import tensorflow as tf
from tensorflow import gfile

from njunmt.utils.global_names import GlobalNames
from njunmt.utils.misc import open_file, close_file
from njunmt.utils.misc import shuffle_data
from njunmt.utils.misc import padding_batch_data


@six.add_metaclass(ABCMeta)
class TextInputter(object):
    """Base class for inputters. """

    def __init__(self,
                 dataset,
                 batch_size=None):
        """ Initializes common attributes of inputters.

        Args:
            dataset: A `Dataset` object.
            batch_size: An integer value indicating the number of
              sentences passed into one step.
        """
        self._dataset = dataset
        self._vocab_source = self._dataset.vocab_source
        self._vocab_target = self._dataset.vocab_target
        self._batch_size = batch_size

    @property
    def input_fields(self):
        """ Returns the dictionary of placeholders. """
        return self._dataset._input_fields

    @abstractmethod
    def make_feeding_data(self, *args, **kwargs):
        """ Processes the data file and return an iterable instance for loop. """
        raise NotImplementedError


class TextLineInputter(TextInputter):
    """ Class for reading in source side lines or target side lines.  """

    def __init__(self,
                 dataset,
                 data_field_name,
                 batch_size):
        """ Initializes the parameters for this inputter.

        Args:
            dataset: A `Dataset` object.
            data_field_name: The attribute name of dataset that has
              access to a data file.
            batch_size: An integer value indicating the number of
              sentences passed into one step. Sentences will be padded by EOS.

        Raises:
            ValueError: if `batch_size` is None, or if `dataset` has no
              attribute named `data_field_name`, or if the attribute
              `data_field_name` has error type (only str and list available).
        """
        super(TextLineInputter, self).__init__(dataset, batch_size)
        if self._batch_size is None:
            raise ValueError("batch_size should be provided.")
        if not hasattr(dataset, data_field_name):
            raise ValueError("dataset object has no attribute named \"{}\""
                             .format(data_field_name))
        self._data_files = getattr(dataset, data_field_name)
        if not (isinstance(self._data_files, str) or isinstance(self._data_files, list)):
            raise ValueError("error type with for attribute \"{}\" of dataset, "
                             "which should be str or list".format(data_field_name))
        if "features" in data_field_name:
            self._vocab = dataset.vocab_source
        else:
            self._vocab = dataset.vocab_target

    def _make_feeding_data_from(self,
                                filename,
                                maximum_line_length=None,
                                maximum_encoded_length=None):
        """ Processes the data file and return an iterable instance for loop.

        Args:
            filename: A specific data file.
            maximum_line_length: The maximum sequence length. If provided,
              sentences exceeding this value will be ignore.
            maximum_encoded_length: The maximum length of symbols (especially
              after BPE is applied). If provided symbols of one sentence exceeding
              this value will be ignore.

        Returns: An iterable instance that packs feeding dictionary
                   for `tf.Session().run` according to the `filename`.
        """
        features = open_file(filename, encoding="utf-8")
        str_buf = []
        ss_buf = []
        for ss in features:
            if maximum_line_length and len(ss.strip().split()) > maximum_line_length:
                continue
            encoded_ss = self._vocab.convert_to_idlist(ss.strip().split(" "))
            if maximum_encoded_length and len(encoded_ss) - 1 > maximum_encoded_length:
                continue
            bpe_ss = self._vocab.bpe_encode(ss.strip())
            str_buf.append(bpe_ss)
            ss_buf.append(encoded_ss)
        close_file(features)
        data = []
        batch_data_idx = 0
        while batch_data_idx < len(ss_buf):
            x, len_x = padding_batch_data(
                ss_buf[batch_data_idx: batch_data_idx + self._batch_size],
                self._vocab.eos_id)
            str_x = str_buf[batch_data_idx: batch_data_idx + self._batch_size]
            batch_data_idx += self._batch_size
            data.append((
                str_x, len_x,
                {self.input_fields[GlobalNames.PH_FEATURE_IDS_NAME]: x,
                 self.input_fields[GlobalNames.PH_FEATURE_LENGTH_NAME]: len_x}))
        return data

    def make_feeding_data(self, maximum_line_length=None, maximum_encoded_length=None):
        """ Processes the data file(s) and return an iterable
        instance for loop.

        Args:
            maximum_line_length: The maximum sequence length. If provided,
              sentences exceeding this value will be ignore.
            maximum_encoded_length: The maximum length of symbols (especially
              after BPE is applied). If provided symbols of one sentence exceeding
              this value will be ignore.

        Returns: An iterable instance or a list of iterable
                   instances according to the `data_field_name`
                   in the constructor.
        """
        if isinstance(self._data_files, list):
            return [self._make_feeding_data_from(filename) for filename in self._data_files]
        return self._make_feeding_data_from(self._data_files)


class ParallelTextInputter(TextInputter):
    """ Class for reading in parallel texts.  """

    def __init__(self,
                 dataset,
                 features_field_name,
                 labels_field_name,
                 batch_size=None,
                 batch_tokens_size=None,
                 shuffle_every_epoch=None,
                 bucketing=True):
        """ Initializes the parameters for this inputter.

        Args:
            dataset: A `Dataset` object.
            features_field_name: The attribute name of dataset that has
              access to a features file.
            labels_field_name: The attribute name of dataset that has
              access to a labels file.
            batch_size: An integer value indicating the number of
              sentences passed into one step. Sentences will be padded by EOS.
            batch_tokens_size: An integer value indicating the number of
              words of each batch. If provided, sentence pairs will be batched
              together by approximate sequence length.
            shuffle_every_epoch: A string type. If provided, use it as postfix
              of shuffled data file name.
            bucketing: Whether to sort the sentences by length of labels.

        Raises:
            ValueError: if both `batch_size` and `batch_tokens_size` are
              not provided, or if `dataset` has no attribute name
              `features_field_name` or `labels_field_name`.

        """
        super(ParallelTextInputter, self).__init__(
            dataset, batch_size)
        self._batch_tokens_size = batch_tokens_size
        self._shuffle_every_epoch = shuffle_every_epoch
        if not hasattr(dataset, features_field_name):
            raise ValueError("dataset object has no attribute named \"{}\""
                             .format(features_field_name))
        if not hasattr(dataset, labels_field_name):
            raise ValueError("dataset object has no attribute named \"{}\""
                             .format(labels_field_name))
        self._features_file = getattr(self._dataset, features_field_name)
        self._labels_file = getattr(self._dataset, labels_field_name)
        self._bucketing = bucketing
        if self._batch_size is None and self._batch_tokens_size is None:
            raise ValueError("Either batch_size or batch_tokens_size should be provided.")
        if (self._batch_size is not None) and (self._batch_tokens_size is not None):
            tf.logging.info("batching data according to batch_tokens_size={}, "
                            "and use batch_size={} as an auxiliary variable.".format(batch_tokens_size, batch_size))
        if batch_tokens_size is None:
            self._cache_size = self._batch_size * 128  # 80 * 128 = 10240
        else:
            self._cache_size = self._batch_tokens_size * 6  # 4096 * 6 := 25000
            if batch_size is None:
                self._batch_size = 32

    def make_feeding_data(self,
                          maximum_features_length=None,
                          maximum_labels_length=None,
                          maximum_encoded_features_length=None,
                          maximum_encoded_labels_length=None):
        """ Processes the data files and return an iterable
              instance for loop.

        Args:
            maximum_features_length: The maximum sequence length of "features" field.
              If provided, sentences exceeding this value will be ignore.
            maximum_labels_length: The maximum sequence length of "labels" field.
              If provided, sentences exceeding this value will be ignore.
            maximum_encoded_features_length: The maximum length of feature symbols (especially
              after BPE is applied) . If provided, the number of symbols of one sentence
              exceeding this value will be ignore.
            maximum_encoded_labels_length: The maximum length of label symbols (especially
              after BPE is applied) . If provided, the number of symbols of one sentence
              exceeding this value will be ignore.

        Returns: An iterable instance or a list of iterable instances.
        """
        if self._features_file is None or self._labels_file is None:
            raise ValueError("Both _features_file and _labels_file should be provided.")
        if isinstance(self._features_file, list):
            return [self._make_feeding_data(f, l, maximum_features_length, maximum_labels_length,
                                            maximum_encoded_features_length, maximum_encoded_labels_length)
                    for f, l in zip(self._features_file, self._labels_file)]
        return self._make_feeding_data(
            self._features_file, self._labels_file,
            maximum_features_length, maximum_labels_length,
            maximum_encoded_features_length, maximum_encoded_labels_length)

    def _make_feeding_data(self,
                           features_file,
                           labels_file,
                           maximum_features_length=None,
                           maximum_labels_length=None,
                           maximum_encoded_features_length=None,
                           maximum_encoded_labels_length=None):
        """ Processes the data files and return an iterable
              instance for loop.

        Args:
            features_file: The path of features file.
            labels_file: The path of labels file.
            maximum_features_length: The maximum sequence length of "features" field.
              If provided, sentences exceeding this value will be ignore.
            maximum_labels_length: The maximum sequence length of "labels" field.
              If provided, sentences exceeding this value will be ignore.
            maximum_encoded_features_length: The maximum length of feature symbols (especially
              after BPE is applied) . If provided, the number of symbols of one sentence
              exceeding this value will be ignore.
            maximum_encoded_labels_length: The maximum length of label symbols (especially
              after BPE is applied) . If provided, the number of symbols of one sentence
              exceeding this value will be ignore.

        Returns: An iterable instance.
        """
        if features_file is None or labels_file is None:
            raise ValueError("Both features_file and labels_file should be provided.")
        line_count = 0
        with gfile.GFile(features_file) as fp:
            for _ in fp:
                line_count += 1
        if line_count > self._cache_size or self._batch_tokens_size is not None:
            return self._BigParallelData(
                self, features_file, labels_file,
                maximum_features_length, maximum_labels_length,
                maximum_encoded_features_length, maximum_encoded_labels_length)
        return self._SmallParallelData(
            features_file, labels_file,
            maximum_features_length, maximum_labels_length,
            maximum_encoded_features_length, maximum_encoded_labels_length)

    def make_eval_feeding_data(self):
        """ Processes the data files and return an iterable instance for loop,
        especially for output_attention when EVAL.

        Returns: An iterable instance or a list of iterable instances.

        """
        if self._features_file is None or self._labels_file is None:
            raise ValueError("Both _features_file and _labels_file should be provided.")
        if isinstance(self._features_file, list):
            return [self._EvalParallelData(f, l)
                    for f, l in zip(self._features_file, self._labels_file)]
        return self._EvalParallelData(
            self._features_file, self._labels_file)

    def _EvalParallelData(self,
                          features_file,
                          labels_file):
        """ Function for reading small scale parallel data for evaluation.

        Args:
            features_file: The path of features file.
            labels_file: The path of labels file.

        Returns: A list of feeding data.
        """
        eval_features = open_file(features_file, encoding="utf-8")
        if gfile.Exists(labels_file):
            eval_labels = open_file(labels_file, encoding="utf-8")
        else:
            eval_labels = open_file(labels_file + "0", encoding="utf-8")
        ss_buf = []
        tt_buf = []
        ss_str_buf = []
        tt_str_buf = []
        for ss, tt in zip(eval_features, eval_labels):
            ss_str = self._vocab_source.bpe_encode(ss.strip()).split(" ")
            tt_str = self._vocab_target.bpe_encode(tt.strip()).split(" ")
            ss_str_buf.append(ss_str)
            tt_str_buf.append(tt_str)
            ss_buf.append(self._vocab_source.convert_to_idlist(ss.strip()))
            tt_buf.append(self._vocab_target.convert_to_idlist(tt.strip()))
        close_file(eval_features)
        close_file(eval_labels)
        if self._bucketing:
            tlen = numpy.array([len(t) for t in tt_buf])
            tidx = tlen.argsort()
            _ss_buf = [ss_buf[i] for i in tidx]
            _tt_buf = [tt_buf[i] for i in tidx]
            _ss_str_buf = [ss_str_buf[i] for i in tidx]
            _tt_str_buf = [tt_str_buf[i] for i in tidx]
            ss_buf = _ss_buf
            tt_buf = _tt_buf
            ss_str_buf = _ss_str_buf
            tt_str_buf = _tt_str_buf
        data = []
        batch_data_idx = 0
        while batch_data_idx < len(ss_buf):
            x, len_x = padding_batch_data(
                ss_buf[batch_data_idx: batch_data_idx + self._batch_size],
                self._vocab_source.eos_id)
            y, len_y = padding_batch_data(
                tt_buf[batch_data_idx: batch_data_idx + self._batch_size],
                self._vocab_target.eos_id)
            data.append((
                ss_str_buf[batch_data_idx: batch_data_idx + self._batch_size],
                tt_str_buf[batch_data_idx: batch_data_idx + self._batch_size], {
                    self.input_fields[GlobalNames.PH_FEATURE_IDS_NAME]: x,
                    self.input_fields[GlobalNames.PH_FEATURE_LENGTH_NAME]: len_x,
                    self.input_fields[GlobalNames.PH_LABEL_IDS_NAME]: y,
                    self.input_fields[GlobalNames.PH_LABEL_LENGTH_NAME]: len_y}))
            batch_data_idx += self._batch_size
        return data

    def _SmallParallelData(self,
                           features_file,
                           labels_file,
                           maximum_features_length=None,
                           maximum_labels_length=None,
                           maximum_encoded_features_length=None,
                           maximum_encoded_labels_length=None):
        """ Function for reading small scale parallel data.

        Args:
            features_file: The path of features file.
            labels_file: The path of labels file.
            maximum_features_length: The maximum sequence length of "features" field.
              If provided, sentences exceeding this value will be ignore.
            maximum_labels_length: The maximum sequence length of "labels" field.
              If provided, sentences exceeding this value will be ignore.
            maximum_encoded_features_length: The maximum length of feature symbols (especially
              after BPE is applied) . If provided, the number of symbols of one sentence
              exceeding this value will be ignore.
            maximum_encoded_labels_length: The maximum length of label symbols (especially
              after BPE is applied) . If provided, the number of symbols of one sentence
              exceeding this value will be ignore.

        Returns: A list of feeding data.
        """
        eval_features = open_file(features_file, encoding="utf-8")
        if gfile.Exists(labels_file):
            eval_labels = open_file(labels_file, encoding="utf-8")
        else:
            eval_labels = open_file(labels_file + "0", encoding="utf-8")
        ss_buf = []
        tt_buf = []
        for ss, tt in zip(eval_features, eval_labels):
            if maximum_features_length and len(ss.strip().split()) > maximum_features_length:
                continue
            if maximum_labels_length and len(tt.strip().split()) > maximum_labels_length:
                continue
            encoded_ss = self._vocab_source.convert_to_idlist(ss.strip().split(" "))
            if maximum_encoded_features_length and len(encoded_ss) - 1 > maximum_encoded_features_length:
                continue
            encoded_tt = self._vocab_target.convert_to_idlist(tt.strip().split(" "))
            if maximum_encoded_labels_length and len(encoded_tt) - 1 > maximum_encoded_labels_length:
                continue
            ss_buf.append(encoded_ss)
            tt_buf.append(encoded_tt)
        close_file(eval_features)
        close_file(eval_labels)
        if self._bucketing:
            tlen = numpy.array([len(t) for t in tt_buf])
            tidx = tlen.argsort()
            _ss_buf = [ss_buf[i] for i in tidx]
            _tt_buf = [tt_buf[i] for i in tidx]
            ss_buf = _ss_buf
            tt_buf = _tt_buf
        data = []
        batch_data_idx = 0
        while batch_data_idx < len(ss_buf):
            x, len_x = padding_batch_data(
                ss_buf[batch_data_idx: batch_data_idx + self._batch_size],
                self._vocab_source.eos_id)
            y, len_y = padding_batch_data(
                tt_buf[batch_data_idx: batch_data_idx + self._batch_size],
                self._vocab_target.eos_id)
            batch_data_idx += self._batch_size
            data.append((len(len_x), {
                self.input_fields[GlobalNames.PH_FEATURE_IDS_NAME]: x,
                self.input_fields[GlobalNames.PH_FEATURE_LENGTH_NAME]: len_x,
                self.input_fields[GlobalNames.PH_LABEL_IDS_NAME]: y,
                self.input_fields[GlobalNames.PH_LABEL_LENGTH_NAME]: len_y}))
        return data

    class _BigParallelData(object):
        """ An iterator class for reading parallel data. """

        def __init__(self,
                     parent,
                     features_file,
                     labels_file,
                     maximum_features_length=None,
                     maximum_labels_length=None,
                     maximum_encoded_features_length=None,
                     maximum_encoded_labels_length=None):
            """ Initializes.

            Args:
                parent: A `ParallelTextInputter` object.
                features_file: The path of features file.
                labels_file: The path of labels file.
                maximum_features_length: The maximum sequence length of "features" field.
                  If provided, sentences exceeding this value will be ignore.
                maximum_labels_length: The maximum sequence length of "labels" field.
                  If provided, sentences exceeding this value will be ignore.
                maximum_encoded_features_length: The maximum length of feature symbols (especially
                  after BPE is applied) . If provided, the number of symbols of one sentence
                  exceeding this value will be ignore.
                maximum_encoded_labels_length: The maximum length of label symbols (especially
                  after BPE is applied) . If provided, the number of symbols of one sentence
                  exceeding this value will be ignore.
            """
            self._parent = parent
            self._features_file = features_file
            self._labels_file = labels_file
            if not gfile.Exists(self._labels_file):
                self._labels_file = self._labels_file + "0"
            self._maximum_features_length = maximum_features_length
            self._maximum_labels_length = maximum_labels_length
            self._maximum_encoded_features_length = maximum_encoded_features_length
            self._maximum_encoded_labels_length = maximum_encoded_labels_length
            if self._parent._shuffle_every_epoch:
                self._shuffle_features_file = self._features_file.strip().split("/")[-1] \
                                              + "." + self._parent._shuffle_every_epoch
                self._shuffle_labels_file = self._labels_file.strip().split("/")[-1] \
                                            + "." + self._parent._shuffle_every_epoch
                self._shuffle()
            self._features = open_file(self._features_file, encoding="utf-8")
            self._labels = open_file(self._labels_file, encoding="utf-8")
            self._features_buffer = []
            self._labels_buffer = []
            self._features_len_buffer = []
            self._labels_len_buffer = []
            self._end_of_data = False

        def __iter__(self):
            return self

        def _reset(self):
            if self._parent._shuffle_every_epoch:
                close_file(self._features)
                close_file(self._labels)
                self._shuffle()
                self._features = open_file(self._features_file, encoding="utf-8")
                self._labels = open_file(self._labels_file, encoding="utf-8")
            self._features.seek(0)
            self._labels.seek(0)

        def __next__(self):
            """ capable for python3
            :return:
            """
            return self.next()

        def _next_features(self):
            ss_tmp = self._features.readline()
            if ss_tmp == "":
                return ""
            ss_tmp = ss_tmp.strip().split(" ")
            if self._maximum_features_length and len(ss_tmp) > self._maximum_features_length:
                return None
            encoded_ss = self._parent._vocab_source.convert_to_idlist(ss_tmp)
            if self._maximum_encoded_features_length and len(
                    encoded_ss) - 1 > self._maximum_encoded_features_length:
                return None
            return encoded_ss

        def _next_labels(self):
            tt_tmp = self._labels.readline()
            if tt_tmp == "":
                return ""
            tt_tmp = tt_tmp.strip().split(" ")
            if self._maximum_labels_length and len(tt_tmp) > self._maximum_labels_length:
                return None
            encoded_tt = self._parent._vocab_target.convert_to_idlist(tt_tmp)
            if self._maximum_encoded_labels_length and len(
                    encoded_tt) - 1 > self._maximum_encoded_labels_length:
                return None
            return encoded_tt

        def next(self):
            if self._end_of_data:
                self._end_of_data = False
                self._reset()
                raise StopIteration

            assert len(self._features_buffer) == len(self._labels_buffer), "Buffer size mismatch"
            if len(self._features_buffer) < self._parent._batch_size:
                cnt = len(self._features_buffer)
                while cnt < self._parent._cache_size:
                    ss = self._next_features()
                    tt = self._next_labels()
                    if ss == "" or tt == "":
                        break
                    if ss is None or tt is None:
                        continue
                    cnt += 1
                    self._features_buffer.append(ss)
                    self._labels_buffer.append(tt)
                if len(self._features_buffer) == 0 or len(self._labels_buffer) == 0:
                    self._end_of_data = False
                    self._reset()
                    raise StopIteration
                if self._parent._bucketing:
                    # sort by len
                    tlen = numpy.array([len(t) for t in self._labels_buffer])
                    tidx = tlen.argsort()
                    _fbuf = [self._features_buffer[i] for i in tidx]
                    _lbuf = [self._labels_buffer[i] for i in tidx]
                    self._features_buffer = _fbuf
                    self._labels_buffer = _lbuf
                self._features_len_buffer = [len(s) for s in self._features_buffer]
                self._labels_len_buffer = [len(t) for t in self._labels_buffer]
            local_batch_size = self._parent._batch_size
            if self._parent._batch_tokens_size is not None:  # batching data by num of tokens
                sum_s = numpy.sum(self._features_len_buffer[: local_batch_size])
                sum_t = numpy.sum(self._labels_len_buffer[: local_batch_size])
                while True:
                    if sum_s >= self._parent._batch_tokens_size or sum_t >= self._parent._batch_tokens_size:
                        break
                    if self._parent._batch_tokens_size - sum_s < 20 or self._parent._batch_tokens_size - sum_t < 20:
                        break
                    if local_batch_size >= len(self._features_len_buffer):
                        break
                    sum_s += self._features_len_buffer[local_batch_size]
                    sum_t += self._labels_len_buffer[local_batch_size]
                    local_batch_size += 1
            features = self._features_buffer[:local_batch_size]
            labels = self._labels_buffer[:local_batch_size]
            if len(features) < local_batch_size:
                del self._features_buffer[:]
                del self._labels_buffer[:]
                del self._features_len_buffer[:]
                del self._labels_len_buffer[:]
            else:
                del self._features_buffer[:local_batch_size]
                del self._labels_buffer[:local_batch_size]
                del self._features_len_buffer[:local_batch_size]
                del self._labels_len_buffer[:local_batch_size]
            if len(features) <= 0 or len(labels) <= 0:
                self._end_of_data = False
                self._reset()
                raise StopIteration
            return len(features), self._make_inputs(features, labels)

        def _make_inputs(self, features, labels):
            x, len_x = padding_batch_data(features, self._parent._vocab_source.eos_id)
            y, len_y = padding_batch_data(labels, self._parent._vocab_target.eos_id)
            return {
                self._parent.input_fields[GlobalNames.PH_FEATURE_IDS_NAME]: x,
                self._parent.input_fields[GlobalNames.PH_FEATURE_LENGTH_NAME]: len_x,
                self._parent.input_fields[GlobalNames.PH_LABEL_IDS_NAME]: y,
                self._parent.input_fields[GlobalNames.PH_LABEL_LENGTH_NAME]: len_y}

        def _shuffle(self):
            """ shuffle features & labels file

            :return:
            """
            tf.logging.info("shuffling data\n\t{} ==> {}\n\t{} ==> {}"
                            .format(self._features_file, self._shuffle_features_file,
                                    self._labels_file, self._shuffle_labels_file))
            shuffle_data([self._features_file, self._labels_file],
                         [self._shuffle_features_file, self._shuffle_labels_file])
            self._features_file = self._shuffle_features_file
            self._labels_file = self._shuffle_labels_file
