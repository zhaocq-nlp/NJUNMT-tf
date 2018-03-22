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
""" Define base experiment class and basic experiment classes. """
import time
from abc import ABCMeta, abstractmethod

import six
import tensorflow as tf

from njunmt.data.dataset import Dataset
from njunmt.data.text_inputter import ParallelTextInputter
from njunmt.data.text_inputter import TextLineInputter
from njunmt.data.vocab import Vocab
from njunmt.inference.decode import evaluate_with_attention
from njunmt.inference.decode import infer
from njunmt.models.model_builder import model_fn
from njunmt.utils.configurable import ModelConfigs
from njunmt.utils.configurable import parse_params
from njunmt.utils.configurable import print_params
from njunmt.utils.configurable import update_eval_metric
from njunmt.utils.configurable import update_infer_params
from njunmt.utils.constants import ModeKeys
from njunmt.utils.metrics import multi_bleu_score_from_file
from njunmt.utils.misc import load_pretrain_model


@six.add_metaclass(ABCMeta)
class Experiment:
    """ Define base experiment class. """

    def __init__(self):
        """Initializes. """
        pass

    @abstractmethod
    def run(self, **kwargs):
        """ Runs the process. """
        raise NotImplementedError

    @staticmethod
    def _build_default_session():
        """ Returns default tf.Session(). """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        config.allow_soft_placement = True
        return tf.Session(config=config)


class TrainingExperiment(Experiment):
    """ Define an experiment for training. """

    def __init__(self, model_configs):
        """ Initializes the training experiment.

        Args:
            model_configs: A dictionary of all configurations.
        """
        super(TrainingExperiment, self).__init__()
        # training options
        training_options = parse_params(
            params=model_configs["train"],
            default_params=self.default_training_options())
        # for datasets
        datasets_params = parse_params(
            params=model_configs["data"],
            default_params=self.default_datasets_params())
        self._model_configs = model_configs
        self._model_configs["train"] = training_options
        self._model_configs["data"] = datasets_params
        print_params("Datasets: ", self._model_configs["data"])
        print_params("Training parameters: ", self._model_configs["train"])
        ModelConfigs.dump(self._model_configs, self._model_configs["model_dir"])

    @staticmethod
    def default_datasets_params():
        """ Returns a dictionary of default "dataset" parameters. """
        return {
            "source_words_vocabulary": None,
            "target_words_vocabulary": None,
            "train_features_file": None,
            "train_labels_file": None,
            "eval_features_file": None,
            "eval_labels_file": None,
            "source_bpecodes": {},
            "target_bpecodes": {}
        }

    @staticmethod
    def default_training_options():
        """ Returns a dictionary of default training options. """
        return {
            "batch_size": 80,
            "batch_tokens_size": None,
            "save_checkpoint_steps": 1000,
            "train_steps": 10000000,
            "eval_steps": 100,
            "pretrain_model": None,
            "reverse_target": False,
            "maximum_features_length": None,
            "maximum_labels_length": None,
            "shuffle_every_epoch": None
        }

    def run(self):
        """ Trains the model. """
        # vocabulary
        self._vocab_source = Vocab(
            filename=self._model_configs["data"]["source_words_vocabulary"],
            bpe_codes=self._model_configs["data"]["source_bpecodes"],
            reverse_seq=False)
        self._vocab_target = Vocab(
            filename=self._model_configs["data"]["target_words_vocabulary"],
            bpe_codes=self._model_configs["data"]["target_bpecodes"],
            reverse_seq=self._model_configs["train"]["reverse_target"])
        # build dataset
        dataset = Dataset(
            self._vocab_source,
            self._vocab_target,
            train_features_file=self._model_configs["data"]["train_features_file"],
            train_labels_file=self._model_configs["data"]["train_labels_file"],
            eval_features_file=self._model_configs["data"]["eval_features_file"],
            eval_labels_file=self._model_configs["data"]["eval_labels_file"])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        estimator_spec = model_fn(model_configs=self._model_configs,
                                  mode=ModeKeys.TRAIN,
                                  dataset=dataset,
                                  name=self._model_configs["problem_name"])
        train_op = estimator_spec.train_op
        hooks = estimator_spec.training_hooks
        # build training session
        sess = tf.train.MonitoredSession(
            session_creator=None,
            hooks=hooks)

        train_text_inputter = ParallelTextInputter(
            dataset,
            "train_features_file",
            "train_labels_file",
            self._model_configs["train"]["batch_size"],
            self._model_configs["train"]["batch_tokens_size"],
            self._model_configs["train"]["shuffle_every_epoch"])
        train_data = train_text_inputter.make_feeding_data(
            maximum_features_length=self._model_configs["train"]["maximum_features_length"],
            maximum_labels_length=self._model_configs["train"]["maximum_labels_length"])
        eidx = 0
        while True:
            if sess.should_stop():
                break
            tf.logging.info("STARTUP Epoch {}".format(eidx))

            for data in train_data:
                if sess.should_stop():
                    break
                sess.run(train_op, feed_dict=data["feed_dict"])
            eidx += 1


class InferExperiment(Experiment):
    """ Define an experiment for inference. """

    def __init__(self, model_configs):
        """ Initializes the inference experiment.

        Args:
            model_configs: A dictionary of all configurations.
        """
        super(InferExperiment, self).__init__()
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
        return {
            "source_words_vocabulary": None,
            "target_words_vocabulary": None,
            "source_bpecodes": {},
            "target_bpecodes": {},
            "batch_size": 32,
            "beam_size": 10,
            "length_penalty": -1.0,
            "maximum_labels_length": 150,
            "delimiter": " ",
            "char_level": False}

    @staticmethod
    def default_inferdata_params():
        """ Returns a dictionary of default infer data parameters. """
        return {
            "features_file": None,
            "output_file": None,
            "labels_file": None,
            "output_attention": False}

    def run(self):
        """Infers data files. """
        # build datasets
        self._vocab_source = Vocab(
            filename=self._model_configs["infer"]["source_words_vocabulary"],
            bpe_codes=self._model_configs["infer"]["source_bpecodes"],
            reverse_seq=False)
        self._vocab_target = Vocab(
            filename=self._model_configs["infer"]["target_words_vocabulary"],
            bpe_codes=self._model_configs["infer"]["target_bpecodes"],
            reverse_seq=self._model_configs["train"]["reverse_target"])
        # build dataset
        dataset = Dataset(
            self._vocab_source,
            self._vocab_target,
            eval_features_file=[p["features_file"] for p
                                in self._model_configs["infer_data"]])

        self._model_configs = update_infer_params(
            self._model_configs,
            beam_size=self._model_configs["infer"]["beam_size"],
            maximum_labels_length=self._model_configs["infer"]["maximum_labels_length"],
            length_penalty=self._model_configs["infer"]["length_penalty"])
        # build model
        estimator_spec = model_fn(model_configs=self._model_configs,
                                  mode=ModeKeys.INFER,
                                  dataset=dataset,
                                  name=self._model_configs["problem_name"])
        predict_op = estimator_spec.predictions

        sess = self._build_default_session()

        text_inputter = TextLineInputter(
            dataset=dataset,
            data_field_name="eval_features_file",
            batch_size=self._model_configs["infer"]["batch_size"])
        # reload
        checkpoint_path = tf.train.latest_checkpoint(self._model_configs["model_dir"])
        if checkpoint_path:
            tf.logging.info("reloading models...")
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
        else:
            raise OSError("File NOT Found. Fail to find checkpoint file from: {}"
                          .format(self._model_configs["model_dir"]))

        tf.logging.info("Start inference.")
        overall_start_time = time.time()

        for infer_data, param in zip(text_inputter.make_feeding_data(),
                                     self._model_configs["infer_data"]):
            tf.logging.info("Infer Source File: {}.".format(param["features_file"]))
            start_time = time.time()
            infer(sess=sess,
                  prediction_op=predict_op,
                  infer_data=infer_data,
                  output=param["output_file"],
                  vocab_source=self._vocab_source,
                  vocab_target=self._vocab_target,
                  delimiter=self._model_configs["infer"]["delimiter"],
                  output_attention=param["output_attention"],
                  tokenize_output=self._model_configs["infer"]["char_level"],
                  verbose=True)
            tf.logging.info("FINISHED {}. Elapsed Time: {}."
                            .format(param["features_file"], str(time.time() - start_time)))
            if param["labels_file"] is not None:
                bleu_score = multi_bleu_score_from_file(
                    hypothesis_file=param["output_file"],
                    references_files=param["labels_file"])
                tf.logging.info("BLEU score (%s): %.2f"
                                % (param["features_file"], bleu_score))
        tf.logging.info("Total Elapsed Time: %s" % str(time.time() - overall_start_time))


class EvalExperiment(Experiment):
    """ Define an experiment for evaluation using loss functions. """

    def __init__(self, model_configs):
        """ Initializes the evaluation experiment.

        Args:
            model_configs: A dictionary of all configurations.
        """
        super(EvalExperiment, self).__init__()
        eval_options = parse_params(
            params=model_configs["eval"],
            default_params=self.default_evaluation_options())
        eval_data = []
        for item in model_configs["eval_data"]:
            eval_data.append(parse_params(
                params=item,
                default_params=self.default_evaldata_params()))
        self._model_configs = model_configs
        self._model_configs["eval"] = eval_options
        self._model_configs["eval_data"] = eval_data
        print_params("Evaluation parameters: ", self._model_configs["eval"])
        print_params("Evaluation datasets: ", self._model_configs["eval_data"])

    @staticmethod
    def default_evaluation_options():
        """ Returns a dictionary of default inference options. """
        return {
            "metric": None,
            "source_words_vocabulary": None,
            "target_words_vocabulary": None,
            "source_bpecodes": {},
            "target_bpecodes": {},
            "batch_size": 32}

    @staticmethod
    def default_evaldata_params():
        """ Returns a dictionary of default infer data parameters. """
        return {
            "features_file": None,
            "labels_file": None,
            "output_attention": False}

    def run(self):
        """Infers data files. """
        # build datasets
        self._vocab_source = Vocab(
            filename=self._model_configs["eval"]["source_words_vocabulary"],
            bpe_codes=self._model_configs["eval"]["source_bpecodes"],
            reverse_seq=False)
        self._vocab_target = Vocab(
            filename=self._model_configs["eval"]["target_words_vocabulary"],
            bpe_codes=self._model_configs["eval"]["target_bpecodes"],
            reverse_seq=self._model_configs["train"]["reverse_target"])
        # build dataset
        dataset = Dataset(
            self._vocab_source,
            self._vocab_target,
            eval_features_file=[p["features_file"] for p
                                in self._model_configs["eval_data"]],
            eval_labels_file=[p["labels_file"] for p
                              in self._model_configs["eval_data"]])

        # update evaluation model config
        self._model_configs, metric_str = update_eval_metric(
            self._model_configs, self._model_configs["eval"]["metric"])
        tf.logging.info("Evaluating using {}".format(metric_str))
        # build model
        estimator_spec = model_fn(model_configs=self._model_configs,
                                  mode=ModeKeys.EVAL,
                                  dataset=dataset,
                                  name=self._model_configs["problem_name"])

        sess = self._build_default_session()
        do_bucketing = (sum([p["output_attention"]
                             for p in self._model_configs["eval_data"]]) == 0)
        text_inputter = ParallelTextInputter(
            dataset=dataset,
            features_field_name="eval_features_file",
            labels_field_name="eval_labels_file",
            batch_size=self._model_configs["eval"]["batch_size"],
            bucketing=do_bucketing)
        # reload
        checkpoint_path = tf.train.latest_checkpoint(self._model_configs["model_dir"])
        if checkpoint_path:
            tf.logging.info("reloading models...")
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
        else:
            raise OSError("File NOT Found. Fail to load checkpoint file from: {}"
                          .format(self._model_configs["model_dir"]))

        tf.logging.info("Start evaluation.")
        overall_start_time = time.time()

        for eval_data, param in zip(text_inputter.make_feeding_data(in_memory=True),
                                    self._model_configs["eval_data"]):
            tf.logging.info("Evaluation Source File: {}.".format(param["features_file"]))
            tf.logging.info("Evaluation Target File: {}.".format(param["labels_file"]))
            start_time = time.time()
            result = evaluate_with_attention(
                sess=sess,
                eval_op=estimator_spec.loss,
                eval_data=eval_data,
                vocab_source=self._vocab_source,
                vocab_target=self._vocab_target,
                attention_op=estimator_spec.predictions \
                    if param["output_attention"] else None,
                output_filename_prefix=param["labels_file"].strip().split("/")[-1])
            tf.logging.info("FINISHED {}. Elapsed Time: {}."
                            .format(param["features_file"], str(time.time() - start_time)))
            tf.logging.info("Evaluation Score ({} on {}): {}"
                            .format(metric_str, param["features_file"], result))
        tf.logging.info("Total Elapsed Time: %s" % str(time.time() - overall_start_time))
