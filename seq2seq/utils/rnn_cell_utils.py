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
import tensorflow as tf
import inspect
import sys

# Import all cell classes from Tensorflow
TF_CELL_CLASSES = [
    x for x in tf.contrib.rnn.__dict__.values()
    if inspect.isclass(x) and issubclass(x, tf.contrib.rnn.RNNCell)
    ]
for cell_class in TF_CELL_CLASSES:
    setattr(sys.modules[__name__], cell_class.__name__, cell_class)


def create_cell(cell_classname, cell_params):
    cell_params = cell_params.copy()

    # Find the cell class, use the in-house implemented LSTMCell & GRUCell
    cell_class = eval(cell_classname)  # find from all CELL NAMES imported from tf.contrib.rnn
    # Make sure additional arguments are valid
    cell_args = set(inspect.getargspec(cell_class.__init__).args[1:])
    new_cell_params = {}
    for key in cell_params.keys():
        if key not in cell_args:
            # raise ValueError(
            tf.logging.info(
                """{} is not a valid argument for {} class. Available arguments
                are: {}""".format(key, cell_class.__name__, cell_args))
        else:
            new_cell_params[key] = cell_params[key]
    # Create cell
    return cell_class(**new_cell_params)


def get_multilayer_rnn_cells(cell_class,
                             cell_params,
                             num_layers=1,
                             dropout_input_keep_prob=1.0,
                             dropout_state_keep_prob=1.0):
    """Creates a new RNN Cell

    Args:
        cell_class: Name of the cell class, e.g. "BasicLSTMCell".
        cell_params: A dictionary of parameters to pass to the cell constructor.
        num_layers: Number of layers. The cell will be wrapped with
          `tf.contrib.rnn.MultiRNNCell`
        dropout_input_keep_prob:
        dropout_state_keep_prob:

    Returns:
        An instance of `tf.contrib.rnn.RNNCell`.
    """
    cells = []
    for _ in range(num_layers):
        cell = create_cell(cell_class, cell_params)
        if dropout_input_keep_prob < 1.0 or dropout_state_keep_prob < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                 input_keep_prob=dropout_input_keep_prob,
                                                 state_keep_prob=dropout_state_keep_prob)
        cells.append(cell)
    # use MultiRNN Cell even its length is 1, for bridge computing
    return tf.contrib.rnn.MultiRNNCell(cells=cells)


def get_condr_rnn_cell(cell_class,
                       cell_params,
                       num_layers=1,
                       dropout_input_keep_prob=1.0,
                       dropout_state_keep_prob=1.0):
    """Creates RNN Cell according to GNMT's architecture

    Args:
        cell_class: Name of the cell class, e.g. "BasicLSTMCell".
        cell_params: A dictionary of parameters to pass to the cell constructor.
        num_layers: Number of layers. The cell will be wrapped with
          `tf.contrib.rnn.MultiRNNCell`
        dropout_input_keep_prob:
        dropout_state_keep_prob:

    Returns:
        An instance of `tf.contrib.rnn.RNNCell`.
    """
    cond_cell = create_cell(cell_class, cell_params)
    if dropout_input_keep_prob < 1.0 or dropout_state_keep_prob < 1.0:
        cond_cell = tf.contrib.rnn.DropoutWrapper(cond_cell,
                                                  input_keep_prob=dropout_input_keep_prob,
                                                  state_keep_prob=dropout_state_keep_prob)
    r_cells = []
    for _ in range(num_layers):
        cell = create_cell(cell_class, cell_params)
        if dropout_input_keep_prob < 1.0 or dropout_state_keep_prob < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                 input_keep_prob=dropout_input_keep_prob,
                                                 state_keep_prob=dropout_state_keep_prob)
        r_cells.append(cell)
    # use a MultiRNNCell as wrapper
    # to deal with hidden state of type tuple
    r_cells = tf.contrib.rnn.MultiRNNCell(cells=r_cells)

    return cond_cell, r_cells
