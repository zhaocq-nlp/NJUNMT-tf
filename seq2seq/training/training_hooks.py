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

import os
import time
import random
import numpy
import copy
import sys
import subprocess
import string
import seq2seq
from seq2seq.utils.global_names import GlobalNames
from seq2seq.utils.utils import dump_model_analysis
from seq2seq.inference.decode import infer
from seq2seq.inference.decode import evaluate
from seq2seq.data.data_iterator import TestTextIterator
from seq2seq.data.data_iterator import EvalTextIterator
from seq2seq.training.summary_writer import SummaryWriter

from tensorflow import gfile
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import meta_graph
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import training_util

from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops.variables import global_variables_initializer


def build_training_hooks(training_configs):
    """ build training hooks

    :param training_configs:
    :return:
    """
    hooks = list()
    hooks.append(
        CheckpointSaverHook(
            checkpoint_dir=training_configs.model_dir,
            training_options=training_configs.training_options,
            training_ops=training_configs.ops,
            checkpoint_basename=GlobalNames.MODEL_CKPT_FILENAME,
            ma_mode=False))
    hooks.append(
        DisplayHook(
            checkpoint_dir=training_configs.model_dir,
            training_ops=training_configs.ops,
            every_n_steps=training_configs.training_options["display_steps"],
            total_steps=training_configs.training_options["train_steps"]))
    for hook_cls, hook_params in training_configs.model_configs["training_hooks"].items():
        hooks.append(eval(hook_cls)(training_configs, training_configs.ops["global_step_tensor"],
                                    **hook_params))
    return hooks


def _build_eval_model(training_configs, model_configs, input_fileds, mode):
    """ build evaluation model

    :param training_configs:
    :param model_configs:
    :param input_fileds:
    :param mode:
    :return:
    """
    eval_model = eval(model_configs["model"])(
        params=model_configs["model_params"],
        mode=mode,
        vocab_source=training_configs.vocab_source,
        vocab_target=training_configs.vocab_target)
    with tf.variable_scope("", reuse=True):
        eval_model_info = eval_model.build(input_fileds)
    return eval_model_info


class SecondOrStepTimer(object):
    """Timer that triggers at most once every N seconds or once every N steps.
    """
    __instances = list()
    __init_triggered_step = 0

    def __init__(self, every_secs=None, every_steps=None, start_at=0):
        self._every_secs = every_secs
        self._every_steps = every_steps
        self._last_triggered_step = (start_at
                                     if SecondOrStepTimer.__init_triggered_step < start_at
                                     else SecondOrStepTimer.__init_triggered_step)
        self._last_triggered_time = time.time()
        self._overall_start_time = time.time()

        SecondOrStepTimer.__instances.append(self)

        if self._every_secs is None and self._every_steps is None:
            raise ValueError("Either every_secs or every_steps should be provided.")
        if (self._every_secs is not None) and (self._every_steps is not None):
            raise ValueError("Can not provide both every_secs and every_steps.")

    def should_trigger_for_step(self, step):
        """Return true if the timer should trigger for the specified step.

        Args:
          step: Training step to trigger on.

        Returns:
          True if the difference between the current time and the time of the last
          trigger exceeds `every_secs`, or if the difference between the current
          step and the last triggered step exceeds `every_steps`. False otherwise.
        """
        if self._last_triggered_step == step:
            return False

        if self._every_secs is not None:
            if time.time() >= self._last_triggered_time + self._every_secs:
                return True

        if self._every_steps is not None:
            if step >= self._last_triggered_step + self._every_steps:
                return True

        return False

    @staticmethod
    def reset_init_triggered_step(step):
        """ reset

        :param step:
        :return:
        """
        SecondOrStepTimer.__init_triggered_step = step
        for ins in SecondOrStepTimer.__instances:
            ins.update_last_triggered_step(step)

    def update_last_triggered_step(self, step):
        """Update the last triggered time and step number.

        Args:
          step: The current step.

        Returns:
          A pair `(elapsed_time, elapsed_steps)`, where `elapsed_time` is the number
          of seconds between the current trigger and the last one (a float), and
          `elapsed_steps` is the number of steps between the current trigger and
          the last one. Both values will be set to `None` on the first trigger.
        """
        if step < self._last_triggered_step:
            return -1., -1, -1.
        current_time = time.time()
        elapsed_secs = current_time - self._last_triggered_time
        elapsed_steps = step - self._last_triggered_step
        elapsed_secs_all = current_time - self._overall_start_time

        self._last_triggered_time = current_time
        self._last_triggered_step = step
        return elapsed_secs, elapsed_steps, elapsed_secs_all

    def last_triggered_step(self):
        return self._last_triggered_step


class LossMetricHook(tf.train.SessionRunHook):
    """ compute loss on devset and half the learning rate according to the dev loss
    """

    def __init__(self, training_configs,
                 global_step_tensor,
                 start_at=0,
                 eval_every_n_steps=100,
                 batch_size=128,
                 dev_target_bpe=None,
                 half_lr_auto=True,
                 half_lr_every_n_steps=None,
                 max_patience=20,
                 min_learning_rate=5e-5,
                 should_summary=True,
                 is_chief=True):
        """ init

        :param training_configs: `namedtuple`, training config
        :param global_step_tensor:  `tensor`, global step
        :param start_at: `int`, starting step
        :param eval_every_n_steps: `int`
        :param batch_size: `int`, decoding batch size
        :param dev_target_bpe: `str`, if provided, use this as the dev target (to deal with bpe)
        :param half_lr_auto: `bool`, whether to half the learning rate
        :param half_lr_every_n_steps: if is not None and can be tranformed to positive integer, half lr every n steps,
            otherwise, use dev loss to half
        :param max_patience: `int`,
        :param min_learning_rate: `float`, minimum learning rate
        :param should_summary:
        :param is_chief:
        :return:
        """
        tf.logging.info("Create LossMetricHook.")
        self._training_configs = training_configs
        self._dev_target_bpe = dev_target_bpe
        self._start_at = start_at
        self._batch_size = batch_size
        self._eval_every_n_steps = eval_every_n_steps
        self._max_patience = max_patience
        self._min_learning_rate = min_learning_rate
        self._global_step_tensor = global_step_tensor

        if half_lr_every_n_steps is not None and half_lr_every_n_steps != "" \
                and int(half_lr_every_n_steps) > 0 and half_lr_auto:
            raise ValueError("in LossMetricHook, you need provide only one way to do learning rate annealing ")
        self._half_lr_auto = half_lr_auto
        self._half_lr_every_n_steps = int(half_lr_every_n_steps)
        self._summary_writer = None
        self._should_summary = should_summary
        self._is_chief = is_chief  # not used now

    def begin(self):
        dev_target = self._training_configs.dev_target
        if self._dev_target_bpe is not None and self._dev_target_bpe != "":
            dev_target = self._dev_target_bpe
        self._dev_data = EvalTextIterator(
            source=self._training_configs.dev_source,
            target=dev_target,
            vocab_source=self._training_configs.vocab_source,
            vocab_target=self._training_configs.vocab_target,
            batch_size=self._batch_size)

        self._input_fields = self._training_configs.input_fields
        # build evaluation model
        tf.logging.info("Building EVAL model in LossMetricHook......")
        info = _build_eval_model(self._training_configs, self._training_configs.model_configs,
                                 self._training_configs.input_fields, tf.contrib.learn.ModeKeys.EVAL)
        self._loss_op = info.loss
        tf.logging.info("Building EVAL model in LossMetricHook......Done")

        # learning rate annealing operation
        with tf.variable_scope("", reuse=True):
            self._lr = tf.get_variable(GlobalNames.LEARNING_RATE_VAR_NAME, shape=(),
                                                   dtype=tf.float32)
        self._half_lr_op = self._lr.assign(
            tf.where(gen_math_ops.less_equal(self._lr / 2., self._min_learning_rate),
                            self._min_learning_rate, self._lr / 2.))
        # aux attributes
        if self._half_lr_auto:
            tf.logging.info("Automatically half the learning according to loss on devset with patience=%d."
                         % self._max_patience)
            self._patience = 0
            self._min_loss = 100000.
        elif isinstance(self._half_lr_every_n_steps, int) \
                and self._half_lr_every_n_steps > 0:
            tf.logging.info("Automatically half the learning every %d steps util %f."
                         % (self._half_lr_every_n_steps, self._min_learning_rate))
            self._half_timer = SecondOrStepTimer(every_steps=self._half_lr_every_n_steps,
                                                 every_secs=None, start_at=self._start_at)
        self._timer = SecondOrStepTimer(every_steps=self._eval_every_n_steps,
                                        every_secs=None)
        if self._should_summary:
            self._summary_writer = SummaryWriter(self._training_configs.model_dir)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        """ compute loss and maybe half the learning rate

        :param run_context:
        :param run_values:
        :return:
        """
        global_step = run_values.results

        if self._timer.should_trigger_for_step(global_step):
            loss = evaluate(sess=run_context.session,
                            input_fields=self._input_fields,
                            eval_op=self._loss_op,
                            evaldata_iterator=self._dev_data)
            tf.logging.info("Evaluating DEVSET: DevLoss=%f  GlobalStep=%d"
                         % (loss, global_step))
            if self._summary_writer is not None:
                self._summary_writer.add_summary("Evaluation/DevLoss", loss, global_step)
            self._timer.update_last_triggered_step(global_step)
            if self._half_lr_auto and global_step >= self._start_at:
                if loss <= self._min_loss:
                    self._min_loss = loss
                    self._patience = 0
                else:
                    self._patience += 1
                    now_lr = self._lr.eval(run_context.session)
                    if self._patience >= self._max_patience and now_lr > self._min_learning_rate:
                        self._patience = 0
                        run_context.session.run(self._half_lr_op)
                        now_lr = self._lr.eval(run_context.session)
                        tf.logging.info(
                            GlobalNames.HOOK_VERBOSE_PREFIX + "in LossMetricHook: Hit maximum patience=%d. HALF THE LEARNING RATE to %f at %d"
                            % (self._max_patience, now_lr, global_step))
                        if now_lr <= self._min_learning_rate:
                            self._max_patience = sys.maxsize  # finish annealing
        if isinstance(self._half_lr_every_n_steps, int) and self._half_lr_every_n_steps > 0 \
                and self._half_timer.should_trigger_for_step(global_step):
            run_context.session.run(self._half_lr_op)
            now_lr = self._lr.eval(run_context.session)
            if now_lr <= self._min_learning_rate:
                self._half_lr_every_n_steps = sys.maxsize
            tf.logging.info(
                GlobalNames.HOOK_VERBOSE_PREFIX + "in LossMetricHook: Hit half_lr_every_n_steps. HALF THE LEARNING RATE to %f at %d"
                % (now_lr, global_step))
            self._half_timer.update_last_triggered_step(global_step)


class BLEUValidationHook(tf.train.SessionRunHook):
    """ decoding and test bleu score
    """

    def __init__(self, training_configs,
                 global_step_tensor,
                 start_at=10000,
                 eval_every_n_steps=1000,
                 batch_size=128,
                 beam_size=5,
                 delimiter=" ",
                 tokenize_output=False,
                 maximum_keep_models=10,
                 tokenize_script="./seq2seq/tools/tokenizeChinese.py",
                 bleu_script="./seq2seq/tools/multi-bleu.perl",
                 should_summary=True,
                 is_chief=True):
        """ init

        :param training_configs: `namedtuple`, trianing configs
        :param global_step_tensor: `tensor`, global step
        :param start_at: `int`, starting step
        :param eval_every_n_steps: `int`
        :param batch_size: `int`
        :param beam_size: `int`
        :param delimiter: `str`, output delimiter
        :param tokenize_output: `bool`, whether to tokenize output
        :param maximum_keep_models: `int`, maximum number of models
        :param tokenize_script:
        :param bleu_script: `str`, BLEU script file
        :param should_summary:
        :param is_chief:
        :return:
        """
        tf.logging.info("Create BLEUValidationHook.")

        self._training_configs = training_configs
        self._checkpoint_dir = training_configs.model_dir
        self._start_at = start_at
        self._eval_every_n_steps = eval_every_n_steps
        self._delimiter = delimiter
        self._tokenize_output = tokenize_output
        self._maximum_keep_models = maximum_keep_models
        self._beam_size = beam_size
        self._batch_size = batch_size
        self._tokenize_script = tokenize_script
        self._bleu_script = bleu_script
        self._global_step_tensor = global_step_tensor
        self._should_summary = should_summary
        self._is_chief = is_chief
        self._summary_writer = None
        self._timer = None

    def begin(self):
        if self._is_chief:
            # dev data
            self._dev_data = TestTextIterator(
                source=self._training_configs.dev_source,
                vocab_source=self._training_configs.vocab_source,
                batch_size=self._batch_size)
            self._dev_target = self._training_configs.dev_target
            self._vocab_target = self._training_configs.vocab_target
            self._input_fields = self._training_configs.input_fields
            # build inference model
            tf.logging.info("Building INFER model in BLEUValidationHook......")
            model_configs = copy.copy(self._training_configs.model_configs)
            model_configs["model_params"]["inference.beam_size"] = self._beam_size
            info = _build_eval_model(self._training_configs, model_configs,
                                     self._training_configs.input_fields, tf.contrib.learn.ModeKeys.INFER)
            self._predict_ops = info.predictions
            tf.logging.info("Building INFER model in BLEUValidationHook......Done")

            # tf.logging BLEUs
            tmp_trans_dir = GlobalNames.TMP_TRANS_DIRNAME_PREFIX + ''.join((''.join(
                random.sample(string.digits + string.ascii_letters, 10))).split())
            if not gfile.Exists(tmp_trans_dir):
                gfile.MakeDirs(tmp_trans_dir)
            self._tmp_trans_file_prefix = os.path.join(tmp_trans_dir, GlobalNames.TMP_TRANS_FILENAME_PREFIX)
            self._best_checkpoints_bleu = list()
            self._best_checkpoints = list()
            if gfile.Exists(GlobalNames.TMP_BLEU_LOG_FILENAME):
                with gfile.GFile(GlobalNames.TMP_BLEU_LOG_FILENAME, "r") as f:
                    self._best_checkpoints_bleu = [float(x) for x in f.readline().strip().split(",")]
                    self._best_checkpoints = [x for x in f.readline().strip().split(",")]

            self._patience = 0
            self._best_bleu_score = -1.

        self._timer = SecondOrStepTimer(every_steps=self._eval_every_n_steps,
                                        every_secs=None, start_at=self._start_at)
        if self._should_summary:
            self._summary_writer = SummaryWriter(self._checkpoint_dir)

    def _check_bleu_script(self):
        """ check whether the BLEU script is correct

        :return:
        """
        if not gfile.Exists(self._bleu_script):
            raise OSError("File not found. Fail to open multi-bleu scrip: %s" % self._bleu_script)
        if gfile.Exists(self._dev_target):
            pseudo_trans = self._dev_target
        else:
            pseudo_trans = self._dev_target + "0"
            if not gfile.Exists(pseudo_trans):
                raise OSError("File not found. Fail to open dev_target: %s or %s"
                              % (self._dev_target, pseudo_trans))
        return int(self._multi_bleu_score(pseudo_trans)) == 100

    def _multi_bleu_score(self, trans_file):
        """ run multi-bleu.perl script

        :param trans_file:
        :return:
        """
        cmd = "perl %s %s < %s" % (self._bleu_script, self._dev_target, trans_file)
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, shell=True)
        popen.wait()
        try:
            bleu_result = popen.stdout.readline().strip()
            bleu = float(bleu_result[7:bleu_result.index(',')])
            stderrs = popen.stderr.readlines()
            if len(stderrs) > 1:
                for line in stderrs:
                    tf.logging.info(line.strip())
        except Exception as e:
            tf.logging.info(e)
            bleu = 0.
        return bleu

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        """ decoding and run

        :param run_context:
        :param run_values:
        :return:
        """
        global_step = run_values.results
        if self._is_chief and self._timer.should_trigger_for_step(global_step):
            self._timer.update_last_triggered_step(global_step)
            # do decode and evaluate bleu score
            output_trans_file = self._tmp_trans_file_prefix + str(global_step)
            samples_src, samples_trg = infer(
                sess=run_context.session,
                input_fields=self._input_fields,
                prediction_op=self._predict_ops,
                testdata_iterator=self._dev_data,
                output=output_trans_file,
                vocab_target=self._vocab_target,
                delimiter=self._delimiter,
                tokenize_script=self._tokenize_output,
                tokenize_output=self._tokenize_output,
                verbose=False)
            # evaluate with BLEU
            bleu = self._multi_bleu_score(output_trans_file)
            if self._summary_writer is not None:
                self._summary_writer.add_summary("Evaluation/DevBLEU", bleu, global_step)
            elapsed_time, _, elapsed_time_all = self._timer.update_last_triggered_step(global_step)
            # tf.logging
            for idx, (s, p) in enumerate(zip(samples_src, samples_trg)):
                tf.logging.info("Sample%d Source: %s" % (idx, s))
                tf.logging.info("Sample%d Prediction: %s\n" % (idx, p))

            tf.logging.info("Evaluating DEVSET: BLEU=%f  GlobalStep=%d   UD %.2f   UDfromStart %.2f"
                         % (bleu, global_step, elapsed_time, elapsed_time_all))
            self._update_bleu_ckpt(bleu, global_step)
            if self._patience >= 30:
                run_context.request_stop()

    def _update_bleu_ckpt(self, bleu, global_step):
        # saving checkpoints with devBLEU
        if bleu >= self._best_bleu_score:
            self._best_bleu_score = bleu
            self._patience = 0
        else:
            self._patience += 1
        if gfile.Exists("%s-%d.meta" % (
                os.path.join(self._checkpoint_dir, GlobalNames.MODEL_CKPT_FILENAME), global_step)):
            if len(self._best_checkpoints) == 0 or bleu > self._best_checkpoints_bleu[0]:
                tarname = GlobalNames.CKPT_TGZ_FILENAME_PREFIX + ("%d.tar.gz" % global_step)
                os.system("tar -zcvf %s %s %s %s %s/*%d*"
                          % (tarname,
                             os.path.join(self._checkpoint_dir, "checkpoint"),
                             os.path.join(self._checkpoint_dir, GlobalNames.MODEL_CONFIG_JSON_FILENAME),
                             os.path.join(self._checkpoint_dir, GlobalNames.MODEL_ANALYSIS_FILENAME),
                             self._checkpoint_dir,
                             global_step))
                self._best_checkpoints_bleu.append(bleu)
                self._best_checkpoints.append(tarname)
                if len(self._best_checkpoints) > self._maximum_keep_models:
                    tidx = numpy.argsort(self._best_checkpoints_bleu)
                    _bleu_buf = [self._best_checkpoints_bleu[i] for i in tidx]
                    _ckpt_buf = [self._best_checkpoints[i] for i in tidx]
                    self._best_checkpoints = _ckpt_buf[1:]
                    self._best_checkpoints_bleu = _bleu_buf[1:]
                    os.system("rm %s" % _ckpt_buf[0])
        else:
            tf.logging.info("Fail to find Checkpoint file of global_step=%d" % global_step)

    def end(self, session):
        if self._is_chief and len(self._best_checkpoints_bleu) > 0:
            with gfile.GFile(GlobalNames.TMP_BLEU_LOG_FILENAME, 'w') as fw:
                fw.write(','.join([str(x) for x in self._best_checkpoints_bleu]) + "\n")
                fw.write(','.join([x for x in self._best_checkpoints]) + "\n")


class CheckpointSaverHook(tf.train.SessionRunHook):
    """Saves checkpoints every N steps or seconds."""

    def __init__(self,
                 checkpoint_dir,
                 training_options,
                 training_ops,
                 saver=None,
                 checkpoint_basename="model.ckpt",
                 ma_mode=False,
                 should_summary=True,
                 is_chief=True):
        """
        Initializes a `CheckpointSaverHook`.

        :param checkpoint_dir: `str`, base directory for the checkpoint files.
        :param training_options: `dict`, training options from input params (tf.FLAGS)
        :param training_ops: `dict`, of tf operations -- loss_op, global_step_tensor
        :param saver: `Saver`
        :param checkpoint_basename:
        :param ma_mode: `bool`, whether is in the model average mode
        :param should_summary:
        :param is_chief:
        :return:
        """

        tf.logging.info("Create CheckpointSaverHook.")
        if saver is None:
            saver = saver_lib._get_saver_or_default()  # pylint: disable=protected-access
        self._saver = saver
        self._checkpoint_dir = checkpoint_dir
        self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)
        # save every n steps
        self._save_every_n_steps = training_options["save_checkpoints_every_n_steps"]
        # variable for session.run
        self._global_step_tensor = training_ops["global_step_tensor"]
        # for after create session
        self._ma_mode = ma_mode
        self._should_summary = should_summary
        self._is_chief = is_chief
        # timer & summary writer
        self._timer = None
        self._summary_writer = None
        if not ma_mode:
            self._global_var_initop = global_variables_initializer()

    def begin(self):
        # dump model details "model_analysis.txt"
        self._curr_iter = 0
        self._timer = SecondOrStepTimer(every_steps=self._save_every_n_steps,
                                        every_secs=None)
        if self._should_summary:
            self._summary_writer = SummaryWriter(self._checkpoint_dir)

    def after_create_session(self, session, coord):
        """ do initialize all variables if not model average mode==True

        :param session:
        :param coord:
        :return:
        """
        checkpoint_path = saver_lib.latest_checkpoint(self._checkpoint_dir)
        if not checkpoint_path and not self._ma_mode:
            tf.logging.info(
                GlobalNames.HOOK_VERBOSE_PREFIX + "in CheckpointSaverHook (after_create_sess): initializing model parameters...")
            session.run(self._global_var_initop)

    def before_run(self, run_context):
        # We do write graph and saver_def at the first call of before_run.
        # We cannot do this in begin, since we let other hooks to change graph and
        # add variables in begin. Graph is finalized after all begin calls.
        if self._is_chief and self._curr_iter == 0:
            training_util.write_graph(
                ops.get_default_graph().as_graph_def(add_shapes=True),
                self._checkpoint_dir,
                "graph.pbtxt")
            dump_model_analysis(self._checkpoint_dir)  # dump model configs
            graph = ops.get_default_graph()
            meta_graph_def = meta_graph.create_meta_graph_def(
                graph_def=graph.as_graph_def(add_shapes=True),
                saver_def=self._saver.saver_def)
            if self._summary_writer is not None:
                self._summary_writer.add_graph(graph)
                self._summary_writer.add_meta_graph(meta_graph_def)
            tf.logging.info(
                GlobalNames.HOOK_VERBOSE_PREFIX + "in CheckpointSaverHook (before_run): dump graph...")
        checkpoint_path = saver_lib.latest_checkpoint(self._checkpoint_dir)
        if checkpoint_path and self._curr_iter == 0:
            # reloading model
            self._saver.restore(run_context.session, checkpoint_path)
            gs = run_context.session.run(self._global_step_tensor)
            tf.logging.info(
                GlobalNames.HOOK_VERBOSE_PREFIX + "in CheckpointSaverHook (after_create_sess): reloading models... and reset global_step=%d" % gs)
            SecondOrStepTimer.reset_init_triggered_step(gs)

        return tf.train.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        global_step = run_values.results
        self._curr_iter = global_step
        if self._is_chief and self._timer.should_trigger_for_step(global_step):
            self._timer.update_last_triggered_step(global_step)
            self._save(global_step, run_context.session)

    def _save(self, step, session):
        """Saves the latest checkpoint."""
        self._saver.save(session, self._save_path, global_step=step)
        tf.logging.info("Saving checkpoints for %d into %s", step, self._save_path)
        if self._summary_writer is not None:
            self._summary_writer.add_session_log(
                tf.SessionLog(
                    status=tf.SessionLog.CHECKPOINT, checkpoint_path=self._save_path),
                step)


class DisplayHook(tf.train.SessionRunHook):
    """Display training info every n steps"""

    def __init__(self,
                 checkpoint_dir,
                 training_ops,
                 every_n_steps=100,
                 total_steps=100000000,
                 should_summary=True,
                 is_chief=True):
        """
        Initializes a `DisplayHook`.

        :param checkpoint_dir: `str`, checkpoint directory
        :param training_ops: `dict`, of tf operations -- loss_op, global_step_tensor
        :param every_n_steps:
        :param total_steps:
        :param should_summary:
        :param is_chief:
        :return:
        """

        tf.logging.info("Create DisplayHook.")
        # save & display steps
        self._every_n_steps = every_n_steps
        self._train_steps = total_steps
        self._should_summary = should_summary
        self._is_chief = is_chief  # not used now
        self._checkpoint_dir = checkpoint_dir
        # variable for session.run
        self._global_step_tensor = training_ops["global_step_tensor"]
        self._training_loss_op = training_ops["loss_op"]
        # timer & summary writer
        self._timer = None
        self._summary_writer = None

    def begin(self):
        self._timer = SecondOrStepTimer(every_steps=self._every_n_steps,
                                        every_secs=None)
        if self._should_summary:
            self._summary_writer = SummaryWriter(self._checkpoint_dir)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(
            {"global_step": self._global_step_tensor,
             "training_loss": self._training_loss_op})

    def after_run(self, run_context, run_values):
        global_step = run_values.results["global_step"]
        training_loss = run_values.results["training_loss"]
        if self._timer.should_trigger_for_step(global_step):
            elapsed_time, elapsed_steps, _ = self._timer.update_last_triggered_step(global_step)
            steps_per_sec = elapsed_steps * 1. / elapsed_time
            secs_per_step = elapsed_time * 1. / elapsed_steps

            tf.logging.info("Update %d \t TrainingLoss=%f   UD %f secs/step"
                         % (global_step, training_loss, secs_per_step))
            if self._summary_writer is not None:
                self._summary_writer.add_summary("global_step/sec", steps_per_sec, global_step)
                self._summary_writer.add_summary("global_step/secs_per_step", secs_per_step, global_step)
                self._summary_writer.add_summary("Loss/training_loss", training_loss, global_step)
        # hit maximum training steps
        if global_step >= self._train_steps:
            run_context.request_stop()
