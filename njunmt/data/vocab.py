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
""" Class and functions for vocabulary. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import collections
from tensorflow import gfile
from njunmt.data.bpe_encdec import BPE
from njunmt.utils.misc import open_file

SpecialVocab = collections.namedtuple("SpecialVocab",
                                      ["UNK", "SEQUENCE_START", "SEQUENCE_END"])


def get_special_vocab(vocabulary_size):
    """ Returns the `SpecialVocab` object for a given vocabulary size.

    Args:
        vocabulary_size: An integer number.

    Returns: The `SpecialVocab` object for a given vocabulary size.
    """
    return SpecialVocab(*range(vocabulary_size, vocabulary_size + 3))


def create_vocabulary_lookup_table_numpy(filename):
    """Creates a lookup table from a vocabulary file.

    Args:
        filename: Path to a vocabulary file containing one word per line.
          Each word is mapped to its line number (starting from 0).

    Returns: A tuple `(word_to_id_mapping, id_to_word_mapping, special_fields)`

    """
    if not gfile.Exists(filename):
        raise ValueError("File does not exist: {}".format(filename))

    # Load vocabulary into memory
    with open_file(filename, encoding="utf-8") as file:
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


class Vocab(object):
    """ Class for vocabulary (feature map) """

    def __init__(self, filename, bpe_codes_file=None, reverse_seq=False):
        """ Initializes the object.

        Args:
            filename: Path to a vocabulary file containing one word per line.
              Each word is mapped to its line number (starting from 0).
            bpe_codes_file: Path to a BPE code file. If provided, do BPE
              before feature mapping.
            reverse_seq: Whether to reverse the sequence when encode the words
              to ids.

        Raises:
            ValueError: if `filename` or `bpe_codes_file` does not exist.
        """
        self.vocab_dict, self.vocab_r_dict, _ = create_vocabulary_lookup_table_numpy(filename)
        self._sos_id = self.vocab_dict["SEQUENCE_START"]
        self._eos_id = self.vocab_dict["SEQUENCE_END"]
        self._unk_id = self.vocab_dict["UNK"]
        self._vocab_size = len(self.vocab_dict)
        self._reverse_seq = reverse_seq
        self._bpe = None
        if bpe_codes_file and not bpe_codes_file == "":
            if not gfile.Exists(bpe_codes_file):
                raise ValueError("bpe_codes_file: {} not exists".format(bpe_codes_file))
            self._bpe = BPE(bpe_codes_file, vocab=filename)

    @property
    def sos_id(self):
        """ Returns the id of the symbol indicating the start of sentence. """
        return self._sos_id

    @property
    def eos_id(self):
        """ Returns the id of the symbol indicating the end of sentence. """
        return self._eos_id

    @property
    def unk_id(self):
        """ Returns the id of the special UNK symbol. """
        return self._unk_id

    @property
    def vocab_size(self):
        """ Returns the size of vocabulary. """
        return self._vocab_size

    def __call__(self, words):
        """ A wrapper method of `convert_to_idlist()` for `map` function,
        because this method is serializable.

        Args:
            words: A list of word tokens.

        Returns: A list of token ids with an extra `eos_id`.
        """
        return self.convert_to_idlist(words)

    def bpe_encode(self, sentence):
        """ Applies BPE encoding.

        Args:
            sentence: A string of sentence or a list of word tokens.

        Returns: The BPE encoding result of the same type as `sentence`.
        """
        if self._bpe:
            return self._bpe.encode(sentence)
        return sentence

    def convert_to_idlist(self, words, n_words=-1):
        """ Maps the sentence into sequence of ids.
              If BPE provided, apply BPE first.

        Args:
            words: A list of word tokens.
            n_words: An integer number. If provided and > 0, token id
              that exceed this value will be mapped into UNK id.

        Returns: A list of token ids with an extra `eos_id`.
        """
        if self._bpe:
            words = self._bpe.encode(words)
        ss = [self.vocab_dict[w] if w in self.vocab_dict else self.unk_id
              for w in words]
        if n_words > 0:
            ss = [w if w < n_words else self.unk_id for w in ss]
        if self._reverse_seq:
            ss = ss[::-1]
        ss += [self.eos_id]
        return ss

    def decorate_with_unk(self, words, unk_symbol="UNK"):
        """ Append (UNK) to the words that are not in the vocabulary.

        Args:
            words: A string or a list of word tokens.
            unk_symbol: A unk symbol.

        Returns: A string or a list of word tokens.
        """
        if isinstance(words, list):
            return [w if w in self.vocab_dict else w + "({})".format(unk_symbol)
                    for w in words]
        elif isinstance(words, six.string_types):
            return " ".join([w if w in self.vocab_dict else w + "({})".format(unk_symbol)
                             for w in words.strip().split()])
        else:
            raise ValueError("Unrecognized type: {}".format(type(words)))

    def convert_to_wordlist(self, pred_ids, bpe_decoding=True, reverse_seq=True):
        """ Converts list of token ids to list of word tokens.

        Args:
            pred_ids: A list of integers indicating token ids.
            bpe_decoding: Whether to recover from BPE. Set to
              false only when using this for displaying attention.
            reverse_seq: Whether to reverse the sequence after transformation.
              Set to false only when using this for displaying attention.

        Returns: A list of word tokens.
        """
        pred_tokens = [self.vocab_r_dict[i] for i in pred_ids]
        if bpe_decoding and self._bpe:
            pred_tokens = self._bpe.decode(pred_tokens)
        if "SEQUENCE_END" in pred_tokens:
            if len(pred_tokens) == 1:
                return ['']
            pred_tokens = pred_tokens[:pred_tokens.index("SEQUENCE_END")]
        if reverse_seq and self._reverse_seq:
            return pred_tokens[::-1]
        return pred_tokens

    def __getitem__(self, item):
        """ Function for operator [].

        Args:
            item: A string or an integer number.

        Returns: The word token if `item` is an integer number,
          or token id if `item` is a string.

        """
        if type(item) is int:
            if item >= self.vocab_size:
                raise ValueError("id {} exceeded the size of vocabulary (size={})".format(item, self.vocab_size))
            return self.vocab_r_dict[item]
        elif isinstance(item, six.string_types):
            return self.vocab_dict[item] if item in self.vocab_dict else self.unk_id
        else:
            raise ValueError("Unrecognized type of item: %s" % str(type(item)))

    @staticmethod
    def equals(vocab1, vocab2):
        """ Compares two `Vocab` objects.

        Args:
            vocab1: A `Vocab` object.
            vocab2: A `Vocab` object.

        Returns: True if two objects are the same, False otherwise.
        """
        if vocab1.vocab_size != vocab2.vocab_size:
            return False
        for key, val in vocab1.vocab_dict.items():
            if key not in vocab2.vocab_dict:
                return False
            elif vocab2[key] != val:
                return False
        return True

    def equals_to(self, vocab):
        """ Compares `self` and `vocab`

        Args:
            vocab: A `Vocab object`

        Returns: True if two objects are the same, False otherwise.
        """
        assert isinstance(vocab, Vocab)
        return Vocab.equals(self, vocab)
