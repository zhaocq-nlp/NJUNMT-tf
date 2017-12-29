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
import sys
import subprocess
import tensorflow as tf


def multi_bleu_score(multibleu_script, ground_truth_file, prediction_file):
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
