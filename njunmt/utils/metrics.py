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
""" Define metric function to evaluation translation results. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import subprocess
import tensorflow as tf

from njunmt.utils.bleu import corpus_bleu
from njunmt.utils.misc import open_file
from njunmt.utils.misc import get_labels_files
from njunmt.utils.misc import deprecated


@deprecated
def multi_bleu_score2(multibleu_script, ground_truth_file, prediction_file):
    """ Runs multi-bleu.perl script and returns the BLEU result.

    Args:
        multibleu_script: The path of multi-bleu.perl.
        ground_truth_file: The reference file.
        prediction_file: The hypothesis file.

    Returns: The BLEU score.
    """
    cmd = "perl {} {} < {}".format(multibleu_script, ground_truth_file, prediction_file)
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, shell=True)
    popen.wait()
    try:
        bleu_result = popen.stdout.readline().strip()
        if sys.version_info < (3,):
            bleu = float(bleu_result[7:bleu_result.index(',')])
        else:
            bleu = float(bleu_result[7:bleu_result.index(b',')])
        stderrs = popen.stderr.readlines()
        if len(stderrs) > 1:
            for line in stderrs:
                tf.logging.info(line.strip())
    except Exception as e:
        tf.logging.info(e)
        bleu = 0.
    return bleu


def multi_bleu_score(hypothesis, references):
    """ Computes corpus-level BLEU.

    Args:
        hypothesis: A 1-d string list.
        references: A 2-d string list, has the same size
          with hypothesis.
    Returns: A float.
    """
    assert (len(hypothesis) == len(references)), "{} vs. {}".format(len(hypothesis), len(references))
    try:
        bleu, _ = corpus_bleu(hypothesis, references)
        bleu = bleu[0]
    except Exception as e:
        tf.logging.info(e)
        bleu = 0.
    return bleu * 100


def multi_bleu_score_from_file(hypothesis_file, references_files):
    """ Computes corpus-level BLEU from hypothesis file
      and reference file(s).

    Args:
        hypothesis_file: A string.
        references_files: A string. The name of reference file or the prefix.
    Returns: A float.
    """
    with open_file(hypothesis_file) as fp:
        hypothesis = fp.readlines()
    references = []
    for ref_file in get_labels_files(references_files):
        with open_file(ref_file) as fp:
            references.append(fp.readlines())
    references = list(map(list, zip(*references)))
    return multi_bleu_score(hypothesis, references)
