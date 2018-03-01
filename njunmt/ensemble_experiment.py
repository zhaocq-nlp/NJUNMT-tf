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
""" Define an experiment for ensemble. """
import time

import tensorflow as tf

from njunmt.data.dataset import Dataset
from njunmt.data.text_inputter import TextLineInputter
from njunmt.data.vocab import Vocab
from njunmt.inference.decode import infer
from njunmt.models.model_builder import model_fn_ensemble
from njunmt.nmt_experiment import Experiment
from njunmt.nmt_experiment import InferExperiment
from njunmt.utils.configurable import parse_params
from njunmt.utils.configurable import print_params
from njunmt.utils.metrics import multi_bleu_score


class EnsembleExperiment(Experiment):
    """ Define an experiment for ensemble model. """
    def __init__(self,
                 model_configs,
                 model_dirs,
                 weight_scheme="average"):
        """ Initializes the ensemble experiment.

        Args:
            model_configs: A dictionary of all configurations.
            model_dirs: A list of model directories (checkpoints).
            weight_scheme: A string, the ensemble weights. See
              `EnsembleModel.get_ensemble_weights()` for more details.
        """
        self._model_dirs = model_dirs
        self._weight_scheme = weight_scheme
        infer_options = parse_params(
            params=model_configs["infer"],
            default_params=self.default_inference_options())
        infer_data = []
        for item in model_configs["infer_data"]:
            infer_data.append(parse_params(
                params=item,
                default_params=self.default_inferdata_params()))
        self._model_configs = model_configs
        self._model_configs["infer"] = infer_options
        self._model_configs["infer_data"] = infer_data
        print_params("Inference parameters: ", self._model_configs["infer"])
        print_params("Inference datasets: ", self._model_configs["infer_data"])

    @staticmethod
    def default_inference_options():
        """ Returns a dictionary of default inference options. """
        return InferExperiment.default_inference_options()

    @staticmethod
    def default_inferdata_params():
        """ Returns a dictionary of default infer data parameters. """
        return {
            "features_file": None,
            "output_file": None,
            "labels_file": None}

    def run(self):
        """ Runs ensemble model. """
        self._vocab_source = Vocab(
            filename=self._model_configs["infer"]["source_words_vocabulary"],
            bpe_codes_file=self._model_configs["infer"]["source_bpecodes"])
        self._vocab_target = Vocab(
            filename=self._model_configs["infer"]["target_words_vocabulary"],
            bpe_codes_file=self._model_configs["infer"]["target_bpecodes"])
        # build dataset
        dataset = Dataset(
            self._vocab_source,
            self._vocab_target,
            eval_features_file=[p["features_file"] for p
                                in self._model_configs["infer_data"]])
        estimator_spec = model_fn_ensemble(
            self._model_dirs, dataset, weight_scheme=self._weight_scheme,
            inference_options=self._model_configs["infer"])
        predict_op = estimator_spec.predictions
        sess = self._build_default_session()
        text_inputter = TextLineInputter(
            dataset=dataset,
            data_field_name="eval_features_file",
            batch_size=self._model_configs["infer"]["batch_size"])
        sess.run(tf.global_variables_initializer())
        tf.logging.info("Start inference.")
        overall_start_time = time.time()

        for feeding_data, param in zip(text_inputter.make_feeding_data(),
                                       self._model_configs["infer_data"]):
            tf.logging.info("Infer Source Features File: {}.".format(param["features_file"]))
            start_time = time.time()
            infer(sess=sess,
                  prediction_op=predict_op,
                  feeding_data=feeding_data,
                  output=param["output_file"],
                  vocab_target=self._vocab_target,
                  delimiter=self._model_configs["infer"]["delimiter"],
                  output_attention=False,
                  tokenize_output=self._model_configs["infer"]["char_level"],
                  tokenize_script=self._model_configs["infer"]["tokenize_script"],
                  verbose=True)
            tf.logging.info("FINISHED {}. Elapsed Time: {}."
                            .format(param["features_file"], str(time.time() - start_time)))
            if param["labels_file"] is not None:
                bleu_score = multi_bleu_score(
                    self._model_configs["infer"]["multibleu_script"],
                    param["labels_file"], param["output_file"])
                tf.logging.info("BLEU score ({}): {}"
                                .format(param["features_file"], bleu_score))
        tf.logging.info("Total Elapsed Time: %s" % str(time.time() - overall_start_time))
