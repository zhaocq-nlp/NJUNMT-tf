import numpy
from njunmt.cells.rnn_cells import LSTMCell
from njunmt.cells.rnn_cells import GRUCell
import tensorflow as tf


def build_inputs(batch_size, dim):
    x = numpy.random.random(size=(batch_size, dim))
    return tf.convert_to_tensor(x, dtype=tf.float32)


class RNNCellTest(tf.test.TestCase):
    batch_size = 7
    dim = 13
    hidden_dim = 17
    beam_size = 3

    input_shape = (batch_size, dim)
    hidden_state_shape = (batch_size, hidden_dim)
    hidden_state_shape_wb = (batch_size * beam_size, hidden_dim)

    def testGRUCell(self):
        input = build_inputs(RNNCellTest.batch_size, RNNCellTest.dim)
        with tf.variable_scope("hehe"):
            my_gru_cell = GRUCell(num_units=RNNCellTest.hidden_dim,
                                  kernel_initializer=tf.constant_initializer(value=0.1))
            my_out, my_state = my_gru_cell(input, my_gru_cell.zero_state(RNNCellTest.batch_size, tf.float32))

        gru_cell = tf.contrib.rnn.GRUCell(RNNCellTest.hidden_dim,
                                          kernel_initializer=tf.constant_initializer(value=0.1))
        out, state = gru_cell(input, gru_cell.zero_state(RNNCellTest.batch_size, tf.float32))
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllEqual(sess.run(my_out), sess.run(out))
            self.assertAllEqual(sess.run(my_state), sess.run(state))


if __name__ == "__main__":
    tf.test.main()
