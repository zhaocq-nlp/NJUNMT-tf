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
""" Define RNN cells. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.rnn import MultiRNNCell


class StackedRNNCell(MultiRNNCell):
    """Extends the Tensorflow MultiRNNCell with name attribute.
    """

    def __init__(self, cells, name="stacked_rnn_cell"):
        """Create a RNN cell composed sequentially of a number of RNNCells.

        Args:
          cells: list of RNNCells that will be composed in this order.
          name: the name of the cell.

        Raises:
          ValueError: if cells is empty (not allowed), or at least one of the cells
            returns a state tuple but the flag `state_is_tuple` is `False`.
        """
        super(StackedRNNCell, self).__init__(cells, state_is_tuple=True)
        self._name = name

    def __call__(self, inputs, state, scope=None):
        """Run this multi-layer cell on inputs, starting from state."""
        return super(StackedRNNCell, self).__call__(
            inputs, state, (scope or self._name))
