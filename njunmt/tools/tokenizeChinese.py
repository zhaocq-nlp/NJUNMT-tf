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
""" The tokenization of Chinese text contains two steps: separate each Chinese characters (by utf-8 encoding); tokenize the non Chinese part (following the mteval script).
Refer to https://github.com/NJUNLP/ZhTokenizer
"""
import re
import sys
import codecs
import six


def is_chinese_char(uchar):
    """ Whether is a chinese character.

    Args:
        uchar: A utf-8 char.

    Returns: True/False.
    """
    if uchar >= u'\u3400' and uchar <= u'\u4db5':  # CJK Unified Ideographs Extension A, release 3.0
        return True
    elif uchar >= u'\u4e00' and uchar <= u'\u9fa5':  # CJK Unified Ideographs, release 1.1
        return True
    elif uchar >= u'\u9fa6' and uchar <= u'\u9fbb':  # CJK Unified Ideographs, release 4.1
        return True
    elif uchar >= u'\uf900' and uchar <= u'\ufa2d':  # CJK Compatibility Ideographs, release 1.1
        return True
    elif uchar >= u'\ufa30' and uchar <= u'\ufa6a':  # CJK Compatibility Ideographs, release 3.2
        return True
    elif uchar >= u'\ufa70' and uchar <= u'\ufad9':  # CJK Compatibility Ideographs, release 4.1
        return True
    elif uchar >= u'\u20000' and uchar <= u'\u2a6d6':  # CJK Unified Ideographs Extension B, release 3.1
        return True
    elif uchar >= u'\u2f800' and uchar <= u'\u2fa1d':  # CJK Compatibility Supplement, release 3.1
        return True
    elif uchar >= u'\uff00' and uchar <= u'\uffef':  # Full width ASCII, full width of English punctuation, half width Katakana, half wide half width kana, Korean alphabet
        return True
    elif uchar >= u'\u2e80' and uchar <= u'\u2eff':  # CJK Radicals Supplement
        return True
    elif uchar >= u'\u3000' and uchar <= u'\u303f':  # CJK punctuation mark
        return True
    elif uchar >= u'\u31c0' and uchar <= u'\u31ef':  # CJK stroke
        return True
    elif uchar >= u'\u2f00' and uchar <= u'\u2fdf':  # Kangxi Radicals
        return True
    elif uchar >= u'\u2ff0' and uchar <= u'\u2fff':  # Chinese character structure
        return True
    elif uchar >= u'\u3100' and uchar <= u'\u312f':  # Phonetic symbols
        return True
    elif uchar >= u'\u31a0' and uchar <= u'\u31bf':  # Phonetic symbols (Taiwanese and Hakka expansion)
        return True
    elif uchar >= u'\ufe10' and uchar <= u'\ufe1f':
        return True
    elif uchar >= u'\ufe30' and uchar <= u'\ufe4f':
        return True
    elif uchar >= u'\u2600' and uchar <= u'\u26ff':
        return True
    elif uchar >= u'\u2700' and uchar <= u'\u27bf':
        return True
    elif uchar >= u'\u3200' and uchar <= u'\u32ff':
        return True
    elif uchar >= u'\u3300' and uchar <= u'\u33ff':
        return True
    else:
        return False


def to_chinese_char(sentences):
    """ Converts a Chinese sentence to character level.

    Args:
        sentences: A utf-8 string or a list of utf-8 strings.

    Returns: A utf-8 string or a list of utf-8 strings.
    """

    def process(sentence):
        sentence = sentence.strip()

        sentence_in_chars = ""
        for c in sentence:
            if is_chinese_char(c):
                sentence_in_chars += " "
                sentence_in_chars += c
                sentence_in_chars += " "
            else:
                sentence_in_chars += c
        sentence = sentence_in_chars

        # tokenize punctuation
        sentence = re.sub(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', r' \1 ', sentence)

        # tokenize period and comma unless preceded by a digit
        sentence = re.sub(r'([^0-9])([\.,])', r'\1 \2 ', sentence)

        # tokenize period and comma unless followed by a digit
        sentence = re.sub(r'([\.,])([^0-9])', r' \1 \2', sentence)

        # tokenize dash when preceded by a digit
        sentence = re.sub(r'([0-9])(-)', r'\1 \2 ', sentence)

        # one space only between words
        sentence = re.sub(r'\s+', r' ', sentence)

        # no leading space
        sentence = re.sub(r'^\s+', r'', sentence)

        # no trailing space
        sentence = re.sub(r'\s+$', r'', sentence)
        return sentence

    if isinstance(sentences, list):
        return [process(s) for s in sentences]
    elif isinstance(sentences, six.string_types):
        return process(sentences)
    else:
        raise ValueError


def tokenize_sgm_file(input_xml_file, output_xml_file):
    """ Converts Chinese sentence from input file to output file (XML file).

    Args:
        input_xml_file: A string.
        output_xml_file: A string.
    """
    file_r = codecs.open(input_xml_file, 'r', encoding="utf-8")  # input file
    file_w = codecs.open(output_xml_file, 'w', encoding="utf-8")  # result file

    for sentence in file_r:
        if sentence.startswith("<seg"):
            start = sentence.find(">") + 1
            end = sentence.rfind("<")
            new_sentence = sentence[:start] + to_chinese_char(sentence[start:end]) + sentence[end:]
        else:
            new_sentence = sentence
        file_w.write(new_sentence)

    file_r.close()
    file_w.close()


def tokenize_plain_file(input_file, output_file):
    """ Converts Chinese sentence from input file to output file (plain text file).

    Args:
        input_file: A string.
        output_file: A string.
    """
    file_r = codecs.open(input_file, 'r', encoding="utf-8")  # input file
    file_w = codecs.open(output_file, 'w', encoding="utf-8")  # result file

    for sentence in file_r:
        file_w.write(to_chinese_char(sentence) + "\n")

    file_r.close()
    file_w.close()


if __name__ == '__main__':
    if sys.argv[1].endswith(".sgm"):
        tokenize_sgm_file(sys.argv[1], sys.argv[2])
    else:
        tokenize_plain_file(sys.argv[1], sys.argv[2])
