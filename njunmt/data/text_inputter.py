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

from njunmt.utils.constants import Constants
from njunmt.utils.constants import concat_name
from njunmt.utils.misc import access_multiple_files
from njunmt.utils.misc import open_file, close_file
from njunmt.utils.misc import shuffle_data
from njunmt.utils.misc import padding_batch_data
from njunmt.utils.expert_utils import repeat_n_times


def read_line_with_filter(
        fp,
        maximum_length=None,
        preprocessing_fn=None):
    """ Reads one line from `fp`, filters by `filter_length` and
      does preprocessing if provided.

    Args:
        fp: A file identifier.
        maximum_length: An integer, the maximum length of one line.
        preprocessing_fn: A callable function.

    Returns: A list.
    """
    line = fp.readline()
    if line == "":
        return ""
    tokens = line.strip()
    if preprocessing_fn:
        tokens = preprocessing_fn(tokens)
    if maximum_length and len(tokens) > maximum_length:
        return None
    return tokens


def do_bucketing(pivot, *args):
    """ Sorts the `pivot` and args by length of `pivot`.

    Args:
        pivot: The pivot.
        args: A list of others.

    Returns: The same as inputs.
    """
    tlen = numpy.array([len(t) for t in pivot])
    tidx = tlen.argsort()
    _pivot = [pivot[i] for i in tidx]
    _args = []
    for ele in args:
        _args.append([ele[i] for i in tidx])
    return _pivot, _args


def pack_feed_dict(name_prefixs, origin_datas, paddings, input_fields):
    """

    Args:
        name_prefixs: A prefix string of a list of strings.
        origin_datas: Data list or a list of data lists.
        paddings: A padding id or a list of padding ids.
        input_fields: A list of input fields dict.

    Returns: A dict for while loop.
    """
    data = dict()
    data["feed_dict"] = dict()

    def map_fn(n, d, p):
        # n: name prefix
        # d: data list
        # p: padding symbol
        data[concat_name(n, Constants.IDS_NAME)] = d
        n_samples = len(d)
        n_devices = len(input_fields)
        n_samples_per_gpu = n_samples // n_devices
        if n_samples % n_devices > 0:
            n_samples_per_gpu += 1

        def _feed_batchs(_start_idx, _inpf):
            if _start_idx * n_samples_per_gpu >= n_samples:
                return 0
            x, x_len = padding_batch_data(
                d[_start_idx * n_samples_per_gpu:(_start_idx + 1) * n_samples_per_gpu], p)
            data["feed_dict"][_inpf[concat_name(n, Constants.IDS_NAME)]] = x
            data["feed_dict"][_inpf[concat_name(n, Constants.LENGTH_NAME)]] = x_len
            return len(x_len)

        parallels = repeat_n_times(
            n_devices, _feed_batchs,
            range(n_devices), input_fields)
        data["feed_dict"]["parallels"] = parallels

    if isinstance(name_prefixs, six.string_types):
        map_fn(name_prefixs, origin_datas, paddings)
    else:
        [map_fn(n, d, p) for n, d, p in zip(name_prefixs, origin_datas, paddings)]
    return data


@six.add_metaclass(ABCMeta)
class TextInputter(object):
    """Base class for inputters. """

    def __init__(self):
        pass

    @abstractmethod
    def make_feeding_data(self, *args, **kwargs):
        """ Processes the data file and return an iterable instance for loop. """
        raise NotImplementedError


class TextLineInputter_new(TextInputter):
    """ Class for reading in lines.  """

    def __init__(self,
                 data_files,
                 vocab,
                 batch_size,
                 maximum_length=None):
        """ Initializes the parameters for this inputter.

        Args:
            filename: A string or a list of string.
            vocab: A `Vocab`.
            batch_size: An integer value indicating the number of
              sentences passed into one step. Sentences will be padded by EOS.
            maximum_length: An integer, the maximum length of the sequence (after encoded).

        Raises:
            ValueError: if `batch_size` is None, or if `filename` not exists.
        """
        super(TextLineInputter_new, self).__init__()
        self._data_files = data_files
        self._batch_size = batch_size
        self._maximum_length = maximum_length
        if self._batch_size is None:
            raise ValueError("batch_size should be provided.")
        self._preprocessing_fn = lambda x: vocab.convert_to_idlist(x)
        self._padding = vocab.pad_id

    def _make_feeding_data_from(self,
                                filename,
                                input_fields,
                                name_prefix):
        """ Processes the data file and return an iterable instance for loop.

        Args:
            filename: A specific data file.
            input_fields: A dict of placeholders.
            name_prefix: A string, the key name prefix for feed_dict.

        Returns: An iterable instance that packs feeding dictionary
                   for `tf.Session().run` according to the `filename`.
        """
        features = open_file(filename, encoding="utf-8")
        ss_buf = []
        while True:
            encoded_ss = read_line_with_filter(
                features, self._maximum_length, self._preprocessing_fn)
            if encoded_ss == "":
                break
            ss_buf.append(encoded_ss)
        close_file(features)
        data = []
        batch_data_idx = 0

        while batch_data_idx < len(ss_buf):
            data.append(pack_feed_dict(
                name_prefixs=name_prefix,
                origin_datas=ss_buf[batch_data_idx: batch_data_idx + self._batch_size],
                paddings=self._padding,
                input_fields=input_fields))
            batch_data_idx += self._batch_size
        return data

    def make_feeding_data(self, input_fields,
                          name_prefix=Constants.FEATURE_NAME_PREFIX):
        """ Processes the data file(s) and return an iterable
        instance for loop.

        Args:
            input_fields: A dict of placeholders.
            name_prefix: A string, the key name prefix for feed_dict.

        Returns: An iterable instance or a list of iterable
                   instances according to the `data_field_name`
                   in the constructor.
        """
        if isinstance(self._data_files, list):
            return [self._make_feeding_data_from(
                filename, input_fields, name_prefix)
                    for filename in self._data_files]
        assert isinstance(self._data_files, six.string_types)
        return self._make_feeding_data_from(
            self._data_files, input_fields, name_prefix)


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
        super(TextLineInputter, self).__init__()
        self._data_field_name = data_field_name
        self._batch_size = batch_size
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
            self._preprocessing_fn = lambda x: dataset.vocab_source.convert_to_idlist(x)
            self._padding = dataset.vocab_source.pad_id
        else:
            self._preprocessing_fn = lambda x: dataset.vocab_target.convert_to_idlist(x)
            self._padding = dataset.vocab_target.pad_id

    def _make_feeding_data_from(self,
                                filename,
                                input_fields,
                                maximum_length=None):
        """ Processes the data file and return an iterable instance for loop.

        Args:
            filename: A specific data file.
            input_fields: A dict of placeholders.
            maximum_length: The maximum length of symbols (especially
              after BPE is applied). If provided symbols of one sentence exceeding
              this value will be ignore.

        Returns: An iterable instance that packs feeding dictionary
                   for `tf.Session().run` according to the `filename`.
        """
        features = open_file(filename, encoding="utf-8")
        ss_buf = []
        encoded_ss = read_line_with_filter(features, maximum_length, self._preprocessing_fn)
        while encoded_ss != "":
            ss_buf.append(encoded_ss)
            encoded_ss = read_line_with_filter(features, maximum_length, self._preprocessing_fn)
        close_file(features)
        data = []
        batch_data_idx = 0
        name_prefix = Constants.FEATURE_NAME_PREFIX \
            if "features" in self._data_field_name else Constants.LABEL_NAME_PREFIX

        while batch_data_idx < len(ss_buf):
            data.append(pack_feed_dict(
                name_prefixs=name_prefix,
                origin_datas=ss_buf[batch_data_idx: batch_data_idx + self._batch_size],
                paddings=self._padding,
                input_fields=input_fields))
            batch_data_idx += self._batch_size
        return data

    def make_feeding_data(self, input_fields, maximum_length=None):
        """ Processes the data file(s) and return an iterable
        instance for loop.

        Args:
            input_fields: A dict of placeholders.
            maximum_length: The maximum length of symbols (especially
              after BPE is applied). If provided symbols of one sentence exceeding
              this value will be ignore.

        Returns: An iterable instance or a list of iterable
                   instances according to the `data_field_name`
                   in the constructor.
        """
        if isinstance(self._data_files, list):
            return [self._make_feeding_data_from(filename, input_fields, maximum_length)
                    for filename in self._data_files]
        return self._make_feeding_data_from(self._data_files, input_fields, maximum_length)


class ParallelTextInputter_new(TextInputter):
    """ Class for reading in parallel texts.  """

    def __init__(self,
                 dataset,
                 maximum_features_length=None,
                 maximum_labels_length=None,
                 batch_size=None,
                 batch_tokens_size=None,
                 shuffle_every_epoch=None,
                 fill_full_batch=False,
                 bucketing=True):
        """ Initializes the parameters for this inputter.

        Args:
            dataset: A `Dataset` object.
            maximum_features_length: The maximum length of symbols (especially
              after BPE is applied). If provided, the number of symbols of
              one sentence exceeding this value will be ignore.
            maximum_labels_length: The maximum length of symbols (especially
              after BPE is applied). If provided, the number of symbols of
              one sentence exceeding this value will be ignore.
            batch_size: An integer value indicating the number of
              sentences passed into one step. Sentences will be padded by EOS.
            batch_tokens_size: An integer value indicating the number of
              words of each batch. If provided, sentence pairs will be batched
              together by approximate sequence length.
            shuffle_every_epoch: A string type. If provided, use it as postfix
              of shuffled data file name.
            fill_full_batch: Whether to ensure each batch of data has `batch_size`
              of datas.
            bucketing: Whether to sort the sentences by length of labels.

        Raises:
            ValueError: if both `batch_size` and `batch_tokens_size` are
              not provided.

        """
        super(ParallelTextInputter_new, self).__init__()
        self._maximum_features_length = maximum_features_length
        self._maximum_labels_length = maximum_labels_length
        self._batch_size = batch_size
        self._batch_tokens_size = batch_tokens_size
        self._shuffle_every_epoch = shuffle_every_epoch
        self._fill_full_batch = fill_full_batch
        self._features_file = access_multiple_files(dataset.features_file)[0]
        self._labels_file = access_multiple_files(dataset.labels_file)[0]
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
        self._features_preprocessing_fn = lambda x: dataset.vocab_source.convert_to_idlist(x)
        self._labels_preprocessing_fn = lambda x: dataset.vocab_target.convert_to_idlist(x)
        self._features_padding = dataset.vocab_source.pad_id
        self._labels_padding = dataset.vocab_target.pad_id

    def _small_parallel_data(self, input_fields):
        """ Function for reading small scale parallel data for evaluation.

        Args:
            input_fields: A dict of placeholders or a list of dicts.

        Returns: A list of feeding data.
        """
        features = open_file(self._features_file, encoding="utf-8")
        labels = open_file(self._labels_file, encoding="utf-8")

        ss_buf = []
        tt_buf = []
        while True:
            ss = read_line_with_filter(features, self._maximum_features_length,
                                       self._features_preprocessing_fn)
            tt = read_line_with_filter(labels, self._maximum_labels_length,
                                       self._labels_preprocessing_fn)
            if ss == "" or tt == "":
                break
            ss_buf.append(ss)
            tt_buf.append(tt)
        close_file(features)
        close_file(labels)
        if self._bucketing:
            tt_buf, ss_buf = do_bucketing(tt_buf, ss_buf)
            ss_buf = ss_buf[0]
        data = []
        batch_data_idx = 0
        while batch_data_idx < len(ss_buf):
            data.append(
                pack_feed_dict(
                    name_prefixs=[Constants.FEATURE_NAME_PREFIX, Constants.LABEL_NAME_PREFIX],
                    origin_datas=[ss_buf[batch_data_idx: batch_data_idx + self._batch_size],
                                  tt_buf[batch_data_idx: batch_data_idx + self._batch_size]],
                    paddings=[self._features_padding, self._labels_padding],
                    input_fields=input_fields))
            batch_data_idx += self._batch_size
        return data

    def make_feeding_data(self,
                          input_fields,
                          in_memory=False):
        """ Processes the data files and return an iterable
              instance for loop.
        Args:
            input_fields: A dict of placeholders or a list of dicts.
            in_memory: Whether to load all data into memory.

        Returns: An iterable instance.
        """
        if in_memory and self._fill_full_batch:
            raise ValueError(
                "`in_memory` option with _SmallParallelData fn now only deal with small evaluation data. "
                "`fill_full_batch` for ParallelTextInputter is available for training data only.")
        if in_memory and self._shuffle_every_epoch:
            raise ValueError(
                "`in_memory` option with _SmallParallelData fn now only deal with small evaluation data. "
                "`shuffle_every_epoch` for ParallelTextInputter is available for training data only.")
        if in_memory:
            return self._small_parallel_data(input_fields)
        return self._BigParallelDataIterator(
            input_fields=input_fields,
            **self.__dict__)

    class _BigParallelDataIterator(object):
        """ An iterator class for reading parallel data. """

        def __init__(self,
                     input_fields,
                     **kwargs):
            """ Initializes.

            Args:
                input_fields: A dict of placeholders or a list of dicts.
                **kwargs: The attributes of the parent ParallelTextInputter.
            """
            for k, v in kwargs.items():
                setattr(self, k, v)

            self._features, self._labels = self._shuffle_and_reopen()
            self._features_buffer = []
            self._labels_buffer = []
            self._features_len_buffer = []
            self._labels_len_buffer = []
            self._end_of_data = False
            self._input_fields = input_fields

        def __iter__(self):
            return self

        def _reset(self):
            self._features, self._labels = self._shuffle_and_reopen()

        def __next__(self):
            """ capable for python3 """
            return self.next()

        def next(self):
            if self._end_of_data:
                self._end_of_data = False
                self._reset()
                raise StopIteration

            assert len(self._features_buffer) == len(self._labels_buffer), "Buffer size mismatch"
            if len(self._features_buffer) < self._batch_size:
                cnt = len(self._features_buffer)
                while cnt < self._cache_size:
                    ss = read_line_with_filter(self._features, self._maximum_features_length,
                                               self._features_preprocessing_fn)
                    tt = read_line_with_filter(self._labels, self._maximum_labels_length,
                                               self._labels_preprocessing_fn)
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
                if self._bucketing:
                    # sort by len
                    self._labels_buffer, self._features_buffer \
                        = do_bucketing(self._labels_buffer, self._features_buffer)
                    self._features_buffer = self._features_buffer[0]
                self._features_len_buffer = [len(s) for s in self._features_buffer]
                self._labels_len_buffer = [len(t) for t in self._labels_buffer]
            if self._fill_full_batch and len(self._features_buffer) < self._batch_size:
                self._end_of_data = False
                self._reset()
                raise StopIteration
            local_batch_size = self._batch_size
            if self._batch_tokens_size is not None:  # batching data by num of tokens
                sum_s = numpy.sum(self._features_len_buffer[: local_batch_size])
                sum_t = numpy.sum(self._labels_len_buffer[: local_batch_size])
                while True:
                    if sum_s >= self._batch_tokens_size or sum_t >= self._batch_tokens_size:
                        break
                    if self._batch_tokens_size - sum_s < 20 or self._batch_tokens_size - sum_t < 20:
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
            ret_data = pack_feed_dict(
                name_prefixs=[Constants.FEATURE_NAME_PREFIX, Constants.LABEL_NAME_PREFIX],
                origin_datas=[features, labels],
                paddings=[self._features_padding, self._labels_padding],
                input_fields=self._input_fields)
            if self._fill_full_batch:
                ret_data["feed_dict"].pop("parallels")
            return ret_data

        def _shuffle_and_reopen(self):
            """ shuffle features & labels file. """
            if self._shuffle_every_epoch:
                if not hasattr(self, "_shuffled_features_file"):
                    self._shuffled_features_file = self._features_file.strip().split("/")[-1] \
                                                   + "." + self._shuffle_every_epoch
                    self._shuffled_labels_file = self._labels_file.strip().split("/")[-1] \
                                                 + "." + self._shuffle_every_epoch

                tf.logging.info("shuffling data\n\t{} ==> {}\n\t{} ==> {}"
                                .format(self._features_file, self._shuffled_features_file,
                                        self._labels_file, self._shuffled_labels_file))
                shuffle_data([self._features_file, self._labels_file],
                             [self._shuffled_features_file, self._shuffled_labels_file])
                self._features_file = self._shuffled_features_file
                self._labels_file = self._shuffled_labels_file
                if hasattr(self, "_features"):
                    close_file(self._features)
                    close_file(self._labels)
            elif hasattr(self, "_features"):
                self._features.seek(0)
                self._labels.seek(0)
                return self._features, self._labels
            return open_file(self._features_file), open_file(self._labels_file)


class ParallelTextInputter(TextInputter):
    """ Class for reading in parallel texts.  """

    def __init__(self,
                 dataset,
                 features_field_name,
                 labels_field_name,
                 batch_size=None,
                 batch_tokens_size=None,
                 shuffle_every_epoch=None,
                 fill_full_batch=False,
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
            fill_full_batch: Whether to ensure each batch of data has `batch_size`
              of datas.
            bucketing: Whether to sort the sentences by length of labels.

        Raises:
            ValueError: if both `batch_size` and `batch_tokens_size` are
              not provided, or if `dataset` has no attribute name
              `features_field_name` or `labels_field_name`.

        """
        super(ParallelTextInputter, self).__init__()
        self._batch_size = batch_size
        self._batch_tokens_size = batch_tokens_size
        self._shuffle_every_epoch = shuffle_every_epoch
        self._fill_full_batch = fill_full_batch
        if not hasattr(dataset, features_field_name):
            raise ValueError("dataset object has no attribute named \"{}\""
                             .format(features_field_name))
        if not hasattr(dataset, labels_field_name):
            raise ValueError("dataset object has no attribute named \"{}\""
                             .format(labels_field_name))
        self._features_file = getattr(dataset, features_field_name)
        self._labels_file = getattr(dataset, labels_field_name)
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
        self._features_preprocessing_fn = lambda x: dataset.vocab_source.convert_to_idlist(x)
        self._labels_preprocessing_fn = lambda x: dataset.vocab_target.convert_to_idlist(x)
        self._features_padding = dataset.vocab_source.pad_id
        self._labels_padding = dataset.vocab_target.pad_id

    def make_feeding_data(self,
                          input_fields,
                          maximum_features_length=None,
                          maximum_labels_length=None,
                          in_memory=False):
        """ Processes the data files and return an iterable
              instance for loop.

        Args:
            input_fields: A dict of placeholders.
            maximum_features_length: The maximum length of symbols (especially
              after BPE is applied). If provided symbols of one sentence exceeding
              this value will be ignore.
            maximum_labels_length: The maximum length of symbols (especially
              after BPE is applied). If provided symbols of one sentence exceeding
              this value will be ignore.
            in_memory: Whether to load all data into memory.

        Returns: An iterable instance or a list of iterable instances.
        """
        if in_memory and self._fill_full_batch:
            raise ValueError(
                "in_memory option with _SmallParallelData fn now only deal with evaluation data. "
                "fill_full_batch for ParallelTextInputter is only for training data.")
        if self._features_file is None or self._labels_file is None:
            raise ValueError("Both _features_file and _labels_file should be provided.")
        if isinstance(self._features_file, list):
            return [self._make_feeding_data(
                f, l, input_fields, maximum_features_length,
                maximum_labels_length, in_memory)
                    for f, l in zip(self._features_file, self._labels_file)]
        return self._make_feeding_data(
            self._features_file, self._labels_file, input_fields,
            maximum_features_length, maximum_labels_length, in_memory)

    def _make_feeding_data(self,
                           features_file,
                           labels_file,
                           input_fields,
                           maximum_features_length=None,
                           maximum_labels_length=None,
                           in_memory=False):
        """ Processes the data files and return an iterable
              instance for loop.

        Args:
            features_file: The path of features file.
            labels_file: The path of labels file.
            input_fields: A dict of placeholders.
            maximum_features_length: The maximum length of feature symbols (especially
              after BPE is applied) . If provided, the number of symbols of one sentence
              exceeding this value will be ignore.
            maximum_labels_length: The maximum length of label symbols (especially
              after BPE is applied) . If provided, the number of symbols of one sentence
              exceeding this value will be ignore.
            in_memory: Whether to load all data into memory.

        Returns: An iterable instance.
        """
        if features_file is None or labels_file is None:
            raise ValueError("Both features_file and labels_file should be provided.")
        if in_memory:
            return self._SmallParallelData(
                features_file, labels_file, input_fields,
                maximum_features_length, maximum_labels_length)
        return self._BigParallelData(
            self, features_file, labels_file,
            input_fields, maximum_features_length,
            maximum_labels_length)

    def _SmallParallelData(self,
                           features_file,
                           labels_file,
                           input_fields,
                           maximum_features_length=None,
                           maximum_labels_length=None):
        """ Function for reading small scale parallel data for evaluation.

        Args:
            features_file: The path of features file.
            labels_file: The path of labels file.
            input_fields: A dict of placeholders.
            maximum_features_length: The maximum length of feature symbols (especially
              after BPE is applied) . If provided, the number of symbols of one sentence
              exceeding this value will be ignore.
            maximum_labels_length: The maximum length of label symbols (especially
              after BPE is applied) . If provided, the number of symbols of one sentence
              exceeding this value will be ignore.

        Returns: A list of feeding data.
        """
        features = open_file(features_file, encoding="utf-8")
        labels = open_file(labels_file[0], encoding="utf-8")

        ss_buf = []
        tt_buf = []
        while True:
            ss = read_line_with_filter(features, maximum_features_length,
                                       self._features_preprocessing_fn)
            tt = read_line_with_filter(labels, maximum_labels_length,
                                       self._labels_preprocessing_fn)
            if ss == "" or tt == "":
                break
            ss_buf.append(ss)
            tt_buf.append(tt)
        close_file(features)
        close_file(labels)
        if self._bucketing:
            tt_buf, ss_buf = do_bucketing(tt_buf, ss_buf)
            ss_buf = ss_buf[0]
        data = []
        batch_data_idx = 0
        while batch_data_idx < len(ss_buf):
            data.append(
                pack_feed_dict(
                    name_prefixs=[Constants.FEATURE_NAME_PREFIX, Constants.LABEL_NAME_PREFIX],
                    origin_datas=[ss_buf[batch_data_idx: batch_data_idx + self._batch_size],
                                  tt_buf[batch_data_idx: batch_data_idx + self._batch_size]],
                    paddings=[self._features_padding, self._labels_padding],
                    input_fields=input_fields))
            batch_data_idx += self._batch_size
        return data

    class _BigParallelData(object):
        """ An iterator class for reading parallel data. """

        def __init__(self,
                     parent,
                     features_file,
                     labels_file,
                     input_fields,
                     maximum_features_length=None,
                     maximum_labels_length=None):
            """ Initializes.

            Args:
                parent: A `ParallelTextInputter` object.
                features_file: The path of features file.
                labels_file: The path of labels file.
                input_fields: A dict of placeholders.
                maximum_features_length: The maximum length of feature symbols (especially
                  after BPE is applied) . If provided, the number of symbols of one sentence
                  exceeding this value will be ignore.
                maximum_labels_length: The maximum length of label symbols (especially
                  after BPE is applied) . If provided, the number of symbols of one sentence
                  exceeding this value will be ignore.
            """
            self._parent = parent
            self._features_file = features_file
            self._labels_file = labels_file[0] if isinstance(labels_file, list) \
                else labels_file
            self._maximum_features_length = maximum_features_length
            self._maximum_labels_length = maximum_labels_length
            self._features, self._labels = self._shuffle_and_reopen()
            self._features_buffer = []
            self._labels_buffer = []
            self._features_len_buffer = []
            self._labels_len_buffer = []
            self._end_of_data = False
            self._input_fields = input_fields

        def __iter__(self):
            return self

        def _reset(self):
            self._features, self._labels = self._shuffle_and_reopen()

        def __next__(self):
            """ capable for python3 """
            return self.next()

        def next(self):
            if self._end_of_data:
                self._end_of_data = False
                self._reset()
                raise StopIteration

            assert len(self._features_buffer) == len(self._labels_buffer), "Buffer size mismatch"
            if len(self._features_buffer) < self._parent._batch_size:
                cnt = len(self._features_buffer)
                while cnt < self._parent._cache_size:
                    ss = read_line_with_filter(self._features, self._maximum_features_length,
                                               self._parent._features_preprocessing_fn)
                    tt = read_line_with_filter(self._labels, self._maximum_labels_length,
                                               self._parent._labels_preprocessing_fn)
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
                    self._labels_buffer, self._features_buffer \
                        = do_bucketing(self._labels_buffer, self._features_buffer)
                    self._features_buffer = self._features_buffer[0]
                self._features_len_buffer = [len(s) for s in self._features_buffer]
                self._labels_len_buffer = [len(t) for t in self._labels_buffer]
            if self._parent._fill_full_batch and len(self._features_buffer) < self._parent._batch_size:
                self._end_of_data = False
                self._reset()
                raise StopIteration
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
            ret_data = pack_feed_dict(
                name_prefixs=[Constants.FEATURE_NAME_PREFIX, Constants.LABEL_NAME_PREFIX],
                origin_datas=[features, labels],
                paddings=[self._parent._features_padding, self._parent._labels_padding],
                input_fields=self._input_fields)
            if self._parent._fill_full_batch:
                ret_data["feed_dict"].pop("parallels")
            return ret_data

        def _shuffle_and_reopen(self):
            """ shuffle features & labels file. """
            if self._parent._shuffle_every_epoch:
                if not hasattr(self, "_shuffled_features_file"):
                    self._shuffled_features_file = self._features_file.strip().split("/")[-1] \
                                                   + "." + self._parent._shuffle_every_epoch
                    self._shuffled_labels_file = self._labels_file.strip().split("/")[-1] \
                                                 + "." + self._parent._shuffle_every_epoch

                tf.logging.info("shuffling data\n\t{} ==> {}\n\t{} ==> {}"
                                .format(self._features_file, self._shuffled_features_file,
                                        self._labels_file, self._shuffled_labels_file))
                shuffle_data([self._features_file, self._labels_file],
                             [self._shuffled_features_file, self._shuffled_labels_file])
                self._features_file = self._shuffled_features_file
                self._labels_file = self._shuffled_labels_file
                if hasattr(self, "_features"):
                    close_file(self._features)
                    close_file(self._labels)
            elif hasattr(self, "_features"):
                self._features.seek(0)
                self._labels.seek(0)
                return self._features, self._labels
            return open_file(self._features_file), open_file(self._labels_file)


if __name__ == "__main__":
    from njunmt.data.dataset import *
    from njunmt.data.vocab import Vocab
    from njunmt.models.sequence_to_sequence import SequenceToSequence
    from njunmt.utils.constants import ModeKeys

    input_fields = SequenceToSequence.create_input_fields(ModeKeys.TRAIN)

    vocab_source = Vocab(
        filename="/Users/zhaocq/Documents/gitdownload/NJUNMT-tf/testdata/vocab.zh")
    vocab_target = Vocab(
        filename="/Users/zhaocq/Documents/gitdownload/NJUNMT-tf/testdata/vocab.en")
    dataset_old = Dataset(
        vocab_source, vocab_target,
        "/Users/zhaocq/Documents/gitdownload/NJUNMT-tf/testdata/toy.zh",
        "/Users/zhaocq/Documents/gitdownload/NJUNMT-tf/testdata/toy.en0")

    dataset_new = Dataset_new(
        vocab_source, vocab_target,
        "/Users/zhaocq/Documents/gitdownload/NJUNMT-tf/testdata/toy.zh",
        "/Users/zhaocq/Documents/gitdownload/NJUNMT-tf/testdata/toy.en0")

    inputter_old = ParallelTextInputter(
        dataset_old, "train_features_file", "train_labels_file",
        32, 500, None, True, True)
    feeding_old = inputter_old.make_feeding_data(input_fields, maximum_features_length=40, maximum_labels_length=40)

    inputter_new = ParallelTextInputter_new(
        dataset_new,
        40, 40,
        32, 500,
        None, True, True)
    feeding_new = inputter_new.make_feeding_data(input_fields)


    def equal_dict(_a, _b):
        for k in _a.keys():
            if isinstance(_a[k], dict):
                equal_dict(_a[k], _b[k])
            else:
                assert type(_a[k]) == type(_b[k])
                assert numpy.array(_a[k]).any() == numpy.array(_b[k]).any(), "hehe"
        return True


    for a, b in zip(feeding_old, feeding_new):
        print(equal_dict(a, b))
