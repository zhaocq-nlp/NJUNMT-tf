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

from tensorflow import gfile
import collections
import sys

SpecialVocab = collections.namedtuple("SpecialVocab",
                                      ["UNK", "SEQUENCE_START", "SEQUENCE_END"])


def get_special_vocab(vocabulary_size):
    """Returns the `SpecialVocab` instance for a given vocabulary size.
    """
    return SpecialVocab(*range(vocabulary_size, vocabulary_size + 3))


def create_vocabulary_lookup_table_numpy(filename):
    """Creates a lookup table for a vocabulary file.

    Args:
        filename: Path to a vocabulary file containg one word per line.
        Each word is mapped to its line number.
    """
    if not gfile.Exists(filename):
        raise ValueError("File does not exist: {}".format(filename))

    # Load vocabulary into memory
    with gfile.GFile(filename) as file:
        vocab = list(line.strip("\n") for line in file)
    vocab_size = len(vocab)

    has_counts = len(vocab[0].split("\t")) == 2
    if has_counts:
        vocab, counts = zip(*[_.split("\t") for _ in vocab])
        counts = [float(_) for _ in counts]
        vocab = list(vocab)
    else:
        counts = [-1. for _ in vocab]

    # Add special vocabulary items
    special_vocab = get_special_vocab(vocab_size)
    vocab += list(special_vocab._fields)
    vocab_size += len(special_vocab)
    counts += [-1. for _ in list(special_vocab._fields)]

    return {v: k for k, v in enumerate(vocab)}, \
           {k: v for k, v in enumerate(vocab)}, \
           special_vocab._fields


def bpe_concat(tokens):
    new_pred_tokens = []
    concat_word = False
    for word in tokens:
        next_concat_word = False
        if word.endswith("@@"):
            word = word[:-2]
            next_concat_word = True
        if concat_word:
            new_pred_tokens[-1] = new_pred_tokens[-1] + word
        else:
            new_pred_tokens.append(word)
        concat_word = next_concat_word
    return new_pred_tokens


class Vocab(object):
    def __init__(self, filename=None, vocab_dict=None, use_bpe=False):
        """

        :param filename: vocab file name
        :param vocab_dict:  or a dict
        """
        assert filename or vocab_dict, "vocab need one init way"
        if filename:
            self.vocab_dict, self.vocab_r_dict, _ = create_vocabulary_lookup_table_numpy(filename)
        else:
            self.vocab_dict = vocab_dict
            for ele in ["UNK", "SEQUENCE_START", "SEQUENCE_END"]:
                idx = len(self.vocab_dict)
                self.vocab_dict[ele] = idx
            self.vocab_r_dict = {idx: word for word, idx in self.vocab_dict.iteritems()}

        self._sos_id = self.vocab_dict["SEQUENCE_START"]
        self._eos_id = self.vocab_dict["SEQUENCE_END"]
        self._unk_id = self.vocab_dict["UNK"]
        self._vocab_size = len(self.vocab_dict)
        self._use_bpe = use_bpe

    @property
    def use_bpe(self):
        return self._use_bpe

    @property
    def sos_id(self):
        return self._sos_id

    @property
    def eos_id(self):
        return self._eos_id

    @property
    def unk_id(self):
        return self._unk_id

    @property
    def vocab_size(self):
        return self._vocab_size

    def convert_to_idlist(self, words, n_words=-1):
        """ encode the words in to word list

        :param words: `list` of `str`
        :param n_words: `int`, unk if word id exceeds n_words
        :return: word id list with an extra eos_id
        """
        ss = [self.vocab_dict[w] if w in self.vocab_dict else self.unk_id
              for w in words]
        if n_words > 0:
            ss = [w if w < n_words else self.unk_id for w in ss]
        ss += [self.eos_id]
        return ss

    def convert_to_wordlist(self, pred_ids, bpe=None):
        """
        convert word id list to word list (according to reverse word dict)
        :param worddict_r:
        :param pred_ids:
        :param bpe:
        :return:
        """
        pred_tokens = [self.vocab_r_dict[i] for i in pred_ids]
        use_bpe = bpe if bpe is not None else self.use_bpe
        if use_bpe:
            pred_tokens = bpe_concat(pred_tokens)
        if 'SEQUENCE_END' in pred_tokens:
            if len(pred_tokens) == 1:
                return ['']
            pred_tokens = pred_tokens[:pred_tokens.index("SEQUENCE_END")]
        if sys.version.startswith("2."):
            return [w.decode("utf-8") for w in pred_tokens]
        else:
            return pred_tokens

    def __getitem__(self, item):
        if type(item) is int:
            if item >= self.vocab_size:
                raise ValueError("id %d exceeded the size of vocabulary (size=%d)" % (item, self.vocab_size))
            return self.vocab_r_dict[item]
        elif type(item) is str:
            return self.vocab_dict[item] if item in self.vocab_dict else self.unk_id
        else:
            raise ValueError("Unrecognized type of item: %s" % str(type(item)))

    @staticmethod
    def equals(vocab1, vocab2):
        if vocab1.vocab_size != vocab2.vocab_size:
            return False
        for key, val in vocab1.vocab_dict.items():
            if key not in vocab2.vocab_dict:
                return False
            elif vocab2[key] != val:
                return False
        return True

    def equals_to(self, vocab):
        return Vocab.equals(self, vocab)
