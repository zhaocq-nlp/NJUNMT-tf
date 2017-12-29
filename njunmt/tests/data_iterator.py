# This file is deprecated
# Copyright 2017 ZhaoChengqi, zhaocq.nlp@gmail.com, Natural Language Processing Group, Nanjing University (2015-2018).
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

import sys
import numpy
import tensorflow as tf
from tensorflow import gfile

from njunmt.utils.misc import open_file, close_file
from njunmt.utils.misc import padding_batch_data
from njunmt.utils.misc import deprecated


def _encode_fun():
    """
    for character level chinese side
    :param l_trg:
    :return:
    """
    if sys.version_info < (3, 0):
        return lambda l_trg: [i.encode('utf-8') for i in list(l_trg)]
    return lambda l_trg: list(l_trg)


encode_fun = _encode_fun()


def shuffle_data(from_binding, to_binding):
    lines_list = []
    fps = []
    fws = []
    for idx in range(len(from_binding)):
        lines_list.append([])
        fps.append(open(from_binding[idx], "r"))
        fws.append(open(to_binding[idx], "w"))

    for zip_lines in zip(*fps):
        for idx in range(len(zip_lines)):
            lines_list[idx].append(zip_lines[idx].strip())
    for fp in fps:
        fp.close()
    rands = numpy.arange(len(lines_list[0]))
    numpy.random.shuffle(rands)
    for i in rands:
        for idx in range(len(lines_list)):
            fws[idx].write(lines_list[idx][i] + "\n")
    for fw in fws:
        fw.close()


@deprecated
class EvalTextIterator:
    def __init__(self, source, target,
                 vocab_source, vocab_target,
                 batch_size=128,
                 n_words_src=-1,
                 n_words_trg=-1):
        # read in batch datas
        f_source = open_file(source)
        if gfile.Exists(target):
            f_target = open_file(target)
        else:
            f_target = open_file(target + "0")

        ss_buf = []
        tt_buf = []
        for ss, tt in zip(f_source, f_target):
            ss = vocab_source.convert_to_idlist(ss.strip().split(), n_words_src)
            tt = vocab_target.convert_to_idlist(tt.strip().split(), n_words_trg)
            ss_buf.append(ss)
            tt_buf.append(tt)
        f_source.close()
        f_target.close()
        tlen = numpy.array([len(t) for t in tt_buf])
        tidx = tlen.argsort()
        _ss_buf = [ss_buf[i] for i in tidx]
        _tt_buf = [tt_buf[i] for i in tidx]
        ss_buf = _ss_buf
        tt_buf = _tt_buf
        self.batch_source_buffer = []
        self.batch_target_buffer = []
        self.batch_data_idx = 0
        self.batch_size = batch_size
        while self.batch_data_idx < len(ss_buf):
            self.batch_source_buffer.append(
                padding_batch_data(ss_buf[self.batch_data_idx: self.batch_data_idx + batch_size], vocab_source.eos_id))
            self.batch_target_buffer.append(
                padding_batch_data(tt_buf[self.batch_data_idx: self.batch_data_idx + batch_size], vocab_target.eos_id))
            self.batch_data_idx += batch_size
        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        self.batch_data_idx = 0

    def __next__(self):
        """
        capable for python3
        :return:
        """
        return self.next()

    def next(self):
        if self.batch_data_idx >= len(self.batch_source_buffer):
            self.reset()
            raise StopIteration

        self.batch_data_idx += 1

        return self.batch_source_buffer[self.batch_data_idx - 1], \
               self.batch_target_buffer[self.batch_data_idx - 1]


@deprecated
class TrainTextIterator:
    """Simple Bitext iterator."""

    def __init__(self, source, target,
                 vocab_source, vocab_target,
                 batch_size=80,
                 maxlen_src=50, maxlen_trg=100,
                 n_words_src=-1, n_words_trg=-1,
                 shuffle_every_epoch=None,
                 shuffle_before_train=None):
        """

        :param source: `str`
        :param target: `str`
        :param vocab_source: `Vocab`
        :param vocab_target: `Vocab`
        :param batch_size: `int`
        :param maxlen_src: `int`
        :param maxlen_trg: `int`
        :param n_words_src: `int`
        :param n_words_trg: `int`
        :param shuffle_every_epoch: if is not None, use it as postfix of shuffled data
        :param shuffle_before_train: if is not None, use it as postfix of shuffled data
        :return:
        """
        if shuffle_before_train:
            tf.logging.info("shuffling data before training\n"
                         "\t%s ==> %s\n\t%s ==> %s"
                         % (source, "./source.shuf." + shuffle_before_train,
                            target, "./target.shuf." + shuffle_before_train))
            shuffle_data([source, target],
                         ["./source.shuf." + shuffle_before_train,
                          "./target.shuf." + shuffle_before_train])
            source = "./source.shuf." + shuffle_before_train
            target = "./target.shuf." + shuffle_before_train
        self.source_file = source
        self.target_file = target

        self.source = open_file(source, encoding='utf-8')
        self.target = open_file(target, encoding='utf-8')

        self.vocab_source = vocab_source
        self.vocab_target = vocab_target

        self.batch_size = batch_size
        self.maxlen_src = maxlen_src
        self.maxlen_trg = maxlen_trg

        self.n_words_src = n_words_src
        self.n_words_trg = n_words_trg

        self.source_buffer = []
        self.target_buffer = []

        self.k = batch_size * 128
        self.end_of_data = False
        self.shuffle_every_epoch = shuffle_every_epoch

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle_every_epoch:
            close_file(self.source)
            close_file(self.target)
            tf.logging.info("shuffling data among epochs")
            shuffle_data([self.source_file, self.target_file],
                         ["./source.shuf." + self.shuffle_every_epoch,
                          "./target.shuf." + self.shuffle_every_epoch])
            self.source = open_file("./source.shuf." + self.shuffle_every_epoch)
            self.target = open_file("./target.shuf." + self.shuffle_every_epoch)
        else:
            self.source.seek(0)
            self.target.seek(0)

    def __next__(self):
        """
        capable for python3
        :return:
        """
        return self.next()

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch'

        if len(self.source_buffer) == 0:
            cnt = 0
            while cnt < self.k:
                ss = self.source.readline()
                if ss == "":
                    break
                tt = self.target.readline()
                if tt == "":
                    break

                ss = ss.strip().split()
                tt = tt.strip().split()
                if len(ss) > self.maxlen_src or len(tt) > self.maxlen_trg:
                    continue

                cnt += 1
                self.source_buffer.append(ss)
                self.target_buffer.append(tt)

            # sort by target buffer
            tlen = numpy.array([len(t) for t in self.target_buffer])
            tidx = tlen.argsort()

            _sbuf = [self.source_buffer[i] for i in tidx]
            _tbuf = [self.target_buffer[i] for i in tidx]

            self.source_buffer = _sbuf
            self.target_buffer = _tbuf

            if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration

        try:
            while True:
                # read source
                try:
                    ss = self.source_buffer.pop(0)
                except IndexError:
                    break
                ss = self.vocab_source.convert_to_idlist(ss, self.n_words_src)

                # read target
                tt = self.target_buffer.pop(0)
                tt = self.vocab_target.convert_to_idlist(tt, self.n_words_trg)

                source.append(ss)
                target.append(tt)

                if len(source) >= self.batch_size or \
                                len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return padding_batch_data(source, self.vocab_source.eos_id), \
               padding_batch_data(target, self.vocab_target.eos_id)


@deprecated
class TestTextIterator:
    def __init__(self, source,
                 vocab_source,
                 batch_size=1,
                 n_words_src=-1):
        # read in batch datas
        f_source = open_file(source)

        ss_buf = []
        ss_str_buf = []
        for ss in f_source:
            # ss_str_buf.append(ss.strip())
            ss_str_buf.append(vocab_source.bpe_encode(ss.strip()))
            ss = vocab_source.convert_to_idlist(ss.strip().split(), n_words_src)
            ss_buf.append(ss)
        f_source.close()

        self.batch_source_buffer = []
        self.batch_source_str_buffer = []

        self.batch_data_idx = 0
        self.batch_size = batch_size
        while self.batch_data_idx < len(ss_buf):
            self.batch_source_buffer.append(
                padding_batch_data(ss_buf[self.batch_data_idx: self.batch_data_idx + batch_size], vocab_source.eos_id))
            self.batch_source_str_buffer.append(
                ss_str_buf[self.batch_data_idx: self.batch_data_idx + batch_size])
            self.batch_data_idx += batch_size
        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        self.batch_data_idx = 0

    def __next__(self):
        """
        capable for python3
        :return:
        """
        return self.next()

    def next(self):
        if self.batch_data_idx >= len(self.batch_source_buffer):
            self.reset()
            raise StopIteration

        self.batch_data_idx += 1
        return self.batch_source_str_buffer[self.batch_data_idx - 1], \
               self.batch_source_buffer[self.batch_data_idx - 1]


if __name__ == "__main__":
    pass
