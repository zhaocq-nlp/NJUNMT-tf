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
""" Utilities. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from collections import namedtuple
import tensorflow as tf

from njunmt.utils.global_names import ModeKeys


class StepTimer(object):
    """Timer that triggers at most once every N steps. """
    __instances = list()
    __program_start_time = time.time()  # overall starting time
    # these two parameters are for DisplayHook
    __last_triggered_start_time = None
    __accumulate_running_time = 0.

    def __init__(self, every_steps, start_at=0):
        """ Initializes the object.

        Args:
            every_steps: A python integer, trigger on this many steps.
            start_at: A python integer, training step to start triggering.
        """
        self._every_steps = every_steps
        self._last_triggered_step = start_at

        StepTimer.__instances.append(self)

    def register_before_run(self):
        """ Registers before session.run(). """
        StepTimer.__last_triggered_start_time = time.time()

    def should_trigger_for_step(self, step):
        """Return true if the timer should trigger for the specified step.

        Accumulates the seconds that training hooks take.

        Args:
            step: A python integer, training step to trigger on.

        Returns: True if the difference between the current
          step and the last triggered step exceeds `every_steps`.
          False otherwise.
        """
        if StepTimer.__last_triggered_start_time is not None:
            StepTimer.__accumulate_running_time += time.time() - StepTimer.__last_triggered_start_time
            StepTimer.__last_triggered_start_time = None

        if self._last_triggered_step == step:
            return False

        if self._every_steps is not None:
            if step >= self._last_triggered_step + self._every_steps:
                return True
        return False

    @staticmethod
    def reset_init_triggered_step(step):
        """ Resets the last triggered step.

        Args:
            step: A python integer, the training step.
        """
        StepTimer.__init_triggered_step = step
        for ins in StepTimer.__instances:
            ins.update_last_triggered_step(step)

    def update_last_triggered_step(self, step):
        """Update the last triggered time and step number.

        Args:
          step: The current step.

        Returns:
          A pair `(elapsed_steps, elapsed_secs_all)`, where `elapsed_steps`
          is the number of steps between the current trigger and the last one,
          and `elapsed_secs_all` is the number of seconds between the current
          trigger and the start time of the program (a float). Both values
          will be set to -1 on the first trigger.
        """
        if step < self._last_triggered_step:
            return -1, -1.
        current_time = time.time()
        elapsed_steps = step - self._last_triggered_step
        elapsed_secs_all = current_time - StepTimer.__program_start_time

        self._last_triggered_time = current_time
        self._last_triggered_step = step
        return elapsed_steps, elapsed_secs_all

    def last_triggered_step(self):
        """ Returns the last triggered step. """
        return self._last_triggered_step

    def get_session_run_time(self):
        """ Returns session running time:
        (total elapsed time - time to apply hooks)"""
        ret = StepTimer.__accumulate_running_time
        StepTimer.__accumulate_running_time = 0.
        return ret


class PadRemover(object):
    """Helper to remove padding from a tensor before sending to the experts.

    The padding is computed for one reference tensor containing the padding mask
    and then can be applied to any other tensor of shape [dim_origin,...].

    Copied from Google's tensor2tensor
    Ex:
        input = [
          [tok1, tok2],
          [tok3, tok4],
          [0, 0],
          [0, 0],
          [tok5, tok6],
          [0, 0],
        ]
        output = [
          [tok1, tok2],
          [tok3, tok4],
          [tok5, tok6],
        ]
    """

    def __init__(self, pad_mask):
        """Compute and store the location of the padding.

        Args:
          pad_mask (tf.Tensor): Reference padding tensor of shape
            [batch_size,length] or [dim_origin] (dim_origin=batch_size*length)
            containing non-zeros positive values to indicate padding location.
        """
        self.nonpad_ids = None
        self.dim_origin = None

        with tf.name_scope("pad_reduce/get_ids"):
            pad_mask = tf.reshape(pad_mask, [-1])  # Flatten the batch
            # nonpad_ids contains coordinates of zeros rows (as pad_mask is
            # float32, checking zero equality is done with |x| < epsilon, with
            # epsilon=1e-9 as standard, here pad_mask only contains positive values
            # so tf.abs would be redundant)
            self.nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))
            self.dim_origin = tf.shape(pad_mask)[:1]

    def remove(self, x):
        """Remove padding from the given tensor.

        Args:
          x (tf.Tensor): of shape [dim_origin,...]

        Returns:
          a tensor of shape [dim_compressed,...] with dim_compressed <= dim_origin
        """
        with tf.name_scope("pad_reduce/remove"):
            x_shape = x.get_shape().as_list()
            x = tf.gather_nd(
                x,
                indices=self.nonpad_ids,
            )
            # This is a hack but for some reason, gather_nd return a tensor of
            # undefined shape, so the shape is set up manually
            x.set_shape([None] + x_shape[1:])
        return x

    def restore(self, x):
        """Add padding back to the given tensor.

        Args:
          x (tf.Tensor): of shape [dim_compressed,...]

        Returns:
          a tensor of shape [dim_origin,...] with dim_compressed >= dim_origin. The
          dim is restored from the original reference tensor
        """
        with tf.name_scope("pad_reduce/restore"):
            x = tf.scatter_nd(
                indices=self.nonpad_ids,
                updates=x,
                shape=tf.concat([self.dim_origin, tf.shape(x)[1:]], axis=0),
            )
        return x


class DecoderOutputRemover(object):
    """ A helper to remove fields from a namedtuple. """
    def __init__(self, mode, all_fields, ignore_fields):
        """ Initializes the fields.

        Only when mode==INFER, apply removes.

        Args:
            mode: A mode.
            all_fields: A list of string, all fields of a namedtuple.
            ignore_fields: A list of string or None, the fields to be removed.
        """
        self._keep_original = True
        if mode == ModeKeys.INFER \
                and ignore_fields is not None \
                and len(ignore_fields) > 0:
            self._keep_original = False
            self._reserved_fields = [x for x in all_fields
                                     if x not in ignore_fields]
            self._new_type = namedtuple("new_type",
                                        self._reserved_fields)

    def apply(self, ins):
        """ Removes elements of a namedtuple instance.

        Args:
            ins: A namedtuple.

        Returns: A new namedtuple.
        """
        if self._keep_original:
            return ins
        return self._new_type(
            **dict(zip(self._reserved_fields,
                       map(lambda i: getattr(ins, i), self._reserved_fields))))