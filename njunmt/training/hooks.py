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
""" Basic running hooks and build_hooks function. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import training_util

from njunmt.utils.constants import Constants
from njunmt.utils.misc import dump_model_analysis
from njunmt.utils.misc import load_pretrain_model
from njunmt.utils.misc import get_saver_or_default
from njunmt.utils.expert_utils import StepTimer
from njunmt.utils.summary_writer import SummaryWriter


def build_hooks(model_configs, distributed_mode=False, is_chief=True):
    """ Builds training hooks.

    Args:
        model_configs: A dictionary of all configurations.
        distributed_mode: Whether is running under a distributed setting.
        is_chief: Whether this is the chief process.

    Returns: A list of `tf.train.SessionRunHook` objects.
    """
    hooks = list()
    if not distributed_mode:
        hooks.append(InitVariablesHook(checkpoint_dir=model_configs["model_dir"]))
    hooks.append(CheckpointSaverHook(
        checkpoint_dir=model_configs["model_dir"],
        save_checkpoint_steps=model_configs["train"]["save_checkpoint_steps"],
        pretrain_model=model_configs["train"]["pretrain_model"],
        problem_name=model_configs["problem_name"],
        model_name=model_configs["model"],
        is_chief=is_chief, do_summary=is_chief))
    hooks.append(DisplayHook(
        checkpoint_dir=model_configs["model_dir"],
        display_steps=model_configs["train"]["eval_steps"],
        maximum_train_steps=model_configs["train"]["train_steps"],
        is_chief=is_chief, do_summary=is_chief))
    # actually, no other hooks now
    if "hooks" in model_configs and isinstance(model_configs["hooks"], list):
        for hook in model_configs["hooks"]:
            hooks.append(
                eval(hook["class"])(
                    is_chief=is_chief, do_summary=is_chief,
                    **hook["params"]))
    return hooks


class InitVariablesHook(tf.train.SessionRunHook):
    """ Define the hook to initialize all global variables."""

    def __init__(self, checkpoint_dir):
        """ Initializes the hook.

        Args:
            checkpoint_dir: A string, the name of the directory that
              checkpoints save to.
        """
        tf.logging.info("Create InitVariablesHook.")
        self._checkpoint_dir = checkpoint_dir
        self._global_var_initop = tf.global_variables_initializer()

    def after_create_session(self, session, coord):
        """ Initializes all global variables after session is created.

        Args:
            session: A TensorFlow Session that has been created.
            coord: A Coordinator object which keeps track of all threads.
        """
        checkpoint_path = saver_lib.latest_checkpoint(self._checkpoint_dir)
        if not checkpoint_path:
            tf.logging.info(
                "InitVariablesHook (after_create_sess): initializing all global variables...")
            session.run(self._global_var_initop)


class CheckpointSaverHook(tf.train.SessionRunHook):
    """ Define the hook that saves checkpoints every N steps."""

    def __init__(self,
                 checkpoint_dir,
                 save_checkpoint_steps=1000,
                 saver=None,
                 pretrain_model=None,
                 problem_name=None,
                 model_name="njunmt.models.SequenceToSequence",
                 do_summary=True,
                 is_chief=True):
        """ Initializes the hook.

        Args:
            checkpoint_dir: A string, base directory for the checkpoint files.
            save_checkpoint_steps: A python integer, save every N steps.
            saver: `Saver` object, used for saving.
            pretrain_model: The pretrained model dir.
            problem_name: A string.
            model_name: The model name.
            do_summary: Whether to save summaries.
            is_chief: Whether this is the chief process.
        """
        tf.logging.info("Create CheckpointSaverHook.")
        if saver is None:
            saver = get_saver_or_default(max_to_keep=8)  # pylint: disable=protected-access
        self._saver = saver
        self._checkpoint_dir = checkpoint_dir
        self._save_path = os.path.join(checkpoint_dir, Constants.MODEL_CKPT_FILENAME)
        self._pretrain_model = pretrain_model
        self._problem_name = problem_name
        self._model_name = model_name
        # save every n steps
        self._save_checkpoint_steps = save_checkpoint_steps
        # variable for session.run
        self._global_step = training_util.get_global_step()
        # for after create session
        self._do_summary = do_summary
        self._is_chief = is_chief
        # timer & summary writer
        self._timer = None
        self._summary_writer = None

    def begin(self):
        """ Creates StepTimer and SummaryWriter. """
        self._first_call = True
        self._timer = StepTimer(every_steps=self._save_checkpoint_steps)
        if self._do_summary:
            self._summary_writer = SummaryWriter(self._checkpoint_dir)
        self._reload_var_ops = None
        if not saver_lib.latest_checkpoint(self._checkpoint_dir) and self._pretrain_model:
            self._reload_var_ops = load_pretrain_model(
                model_name=self._model_name,
                pretrain_model_dir=self._pretrain_model,
                problem_name=self._problem_name)

    def after_create_session(self, session, coord):
        checkpoint_path = saver_lib.latest_checkpoint(self._checkpoint_dir)
        if checkpoint_path:
            # reloading model
            self._saver.restore(session, checkpoint_path)
            gs = session.run(self._global_step)
            tf.logging.info(
                "CheckpointSaverHook (after_create_session): reloading models and reset global_step={}".format(gs))
            StepTimer.reset_init_triggered_step(gs)
        elif self._reload_var_ops:
            tf.logging.info("Assign all variables with pretrained variables.")
            session.run(self._reload_var_ops)

    def before_run(self, run_context):
        """ Dumps graphs and loads checkpoint if there exits.

        Called before each call to run().

        Args:
            run_context: A `SessionRunContext` object.

        Returns: A `SessionRunArgs` object containing global_step.
        """
        # We do write graph and saver_def at the first call of before_run.
        # We cannot do this in begin, since we let other hooks to change graph and
        # add variables in begin. Graph is finalized after all begin calls.
        if self._is_chief and self._first_call:
            training_util.write_graph(
                ops.get_default_graph().as_graph_def(add_shapes=True),
                self._checkpoint_dir,
                "graph.pbtxt")
            # dump model details "model_analysis.txt"
            dump_model_analysis(self._checkpoint_dir)  # dump model configs
            graph = ops.get_default_graph()
            meta_graph_def = meta_graph.create_meta_graph_def(
                graph_def=graph.as_graph_def(add_shapes=True),
                saver_def=self._saver.saver_def)
            if self._summary_writer is not None:
                self._summary_writer.add_graph(graph)
                self._summary_writer.add_meta_graph(meta_graph_def)
            tf.logging.info("CheckpointSaverHook (before_run): dump graph...")
        self._first_call = False
        self._timer.register_before_run()
        return tf.train.SessionRunArgs(self._global_step)

    def after_run(self, run_context, run_values):
        """ Checks running steps and save checkpoints.

        Args:
            run_context: A `SessionRunContext` object.
            run_values: A SessionRunValues object.
        """
        global_step = run_values.results
        if self._is_chief and self._timer.should_trigger_for_step(global_step):
            self._timer.update_last_triggered_step(global_step)
            self._save(global_step, run_context.session)

    def _save(self, step, session):
        """ Saves checkpoints.

        Args:
            step: A python integer, running step.
            session: A TensorFlow Session.
        """
        """Saves the latest checkpoint."""
        self._saver.save(session, self._save_path, global_step=step)
        tf.logging.info("Saving checkpoints for {} into {}".format(step, self._save_path))
        if self._summary_writer is not None:
            self._summary_writer.add_session_log(
                SessionLog(
                    status=SessionLog.CHECKPOINT, checkpoint_path=self._save_path),
                step)


class DisplayHook(tf.train.SessionRunHook):
    """ Define the hook to display training loss, training speed and
    learning rate every n steps and determine when to stop. """

    def __init__(self,
                 checkpoint_dir,
                 display_steps=100,
                 maximum_train_steps=None,
                 do_summary=True,
                 is_chief=True):
        """ Initializes the hook.

        Args:
            checkpoint_dir: A string, base directory for the checkpoint files.
            display_steps: A python integer, display every N steps.
            maximum_train_steps: A python integer, the maximum training steps.
            do_summary: Whether to save summaries when display.
            is_chief: Whether this is the chief process.do_summary:
        """

        tf.logging.info("Create DisplayHook.")
        self._checkpoint_dir = checkpoint_dir
        # display steps
        self._display_steps = display_steps
        self._maximum_train_steps = maximum_train_steps
        self._do_summary = do_summary
        self._is_chief = is_chief  # not used now

        # display values
        global_step = training_util.get_global_step()
        display_keys = ops.get_collection(Constants.DISPLAY_KEY_COLLECTION_NAME)
        display_values = ops.get_collection(Constants.DISPLAY_VALUE_COLLECTION_NAME)
        self._display_args = dict(zip(display_keys, display_values))
        self._display_args["global_step"] = global_step
        # timer & summary writer
        self._timer = None
        self._summary_writer = None

    def begin(self):
        """ Creates StepTimer and SummaryWriter. """
        self._timer = StepTimer(every_steps=self._display_steps)
        if self._do_summary:
            self._summary_writer = SummaryWriter(self._checkpoint_dir)

    def before_run(self, run_context):
        """ Dumps graphs and loads checkpoint if there exits.

        Called before each call to run().

        Args:
            run_context: A `SessionRunContext` object.

        Returns: A `SessionRunArgs` object containing global_step and
          arguments to be displayed.
        """
        self._timer.register_before_run()
        return tf.train.SessionRunArgs(self._display_args)

    def after_run(self, run_context, run_values):
        """ Checks running steps and print args.

        Also checks the maximum training steps to raise stop request.

        Args:
            run_context: A `SessionRunContext` object.
            run_values: A SessionRunValues object.
        """
        global_step = run_values.results.pop("global_step")
        if self._timer.should_trigger_for_step(global_step):

            training_loss = run_values.results[Constants.TRAIN_LOSS_KEY_NAME]
            elapsed_steps, _ = self._timer.update_last_triggered_step(global_step)
            session_run_time = self._timer.get_session_run_time()
            steps_per_sec = elapsed_steps * 1. / session_run_time
            secs_per_step = session_run_time * 1. / elapsed_steps

            tf.logging.info("Update %d \t TrainingLoss=%f   UD %f secs/step"
                            % (global_step, training_loss, secs_per_step))
            if self._summary_writer is not None:
                self._summary_writer.add_summary("global_step/sec", steps_per_sec, global_step)
                self._summary_writer.add_summary("global_step/secs_per_step", secs_per_step, global_step)
                for k, v in run_values.results.items():
                    self._summary_writer.add_summary(k, v, global_step)
        # hit maximum training steps
        if self._maximum_train_steps and global_step >= self._maximum_train_steps:
            tf.logging.info("Training maximum steps. maximum_train_step={}".format(self._maximum_train_steps))
            run_context.request_stop()
