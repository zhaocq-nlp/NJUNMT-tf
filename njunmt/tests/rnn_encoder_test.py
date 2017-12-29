import numpy
import tensorflow as tf
from njunmt.encoders import rnn_encoder
from tensorflow.python.util import nest


def build_inputs(batch_size, max_len, dim):
    x = numpy.random.random(size=(batch_size, max_len, dim))
    x_len = numpy.random.randint(low=1, high=max_len, size=(batch_size,))
    return tf.convert_to_tensor(x, dtype=tf.float32), \
           tf.convert_to_tensor(x_len, dtype=tf.int32)


def flatten_final_states(final_states):
    if isinstance(final_states, dict):
        return nest.flatten(list(final_states.values()))
    return nest.flatten(final_states)


def build_state(shape, is_lstm, batch_to_none=True):
    state = tf.convert_to_tensor(numpy.zeros(shape=shape))
    if batch_to_none:
        state.set_shape((None,) + tuple(shape)[1:])
    if is_lstm:
        return tf.contrib.rnn.LSTMStateTuple(state, state)
    return state


class RNNEncoderTest(tf.test.TestCase):
    batch_size = 7
    max_len = 11
    dim = 17
    hidden_dim = 5

    input_shape = (batch_size, max_len, dim)
    hidden_state_shape = (batch_size, hidden_dim)
    uni_context_shape = (batch_size, max_len, hidden_dim)
    bi_context_shape = (batch_size, max_len, hidden_dim * 2)

    little_params1 = {"rnn_cell": {
        "cell_class": "LSTMCell",
        "cell_params": {
            "num_units": hidden_dim,
            "layer_norm": False,
            "dropout_input_keep_prob": 0.9,
            "dropout_state_keep_prob": 1.0
        },
        "residual_connections": False,
        "num_layers": 1
    }}

    little_params2 = {"rnn_cell": {
        "cell_class": "LSTMCell",
        "cell_params": {
            "num_units": hidden_dim,
            "layer_norm": True,
            "dropout_input_keep_prob": 1.0,
            "dropout_state_keep_prob": 1.0
        },
        "residual_connections": False,
        "num_layers": 3
    }}

    little_params3 = {"rnn_cell": {
        "cell_class": "GRUCell",
        "cell_params": {
            "num_units": hidden_dim,
            "layer_norm": True,
            "dropout_input_keep_prob": 1.0,
            "dropout_state_keep_prob": 1.0
        },
        "residual_connections": False,
        "num_layers": 3
    }}

    def testStackUnidirectionalRNNEncoder1(self):
        encoder = rnn_encoder.UnidirectionalRNNEncoder(RNNEncoderTest.little_params1,
                                                       tf.contrib.learn.ModeKeys.TRAIN)
        encoder_output = encoder.encode(*build_inputs(*RNNEncoderTest.input_shape))
        self.assertAllEqual(encoder_output.outputs.shape,
                            tf.TensorShape(RNNEncoderTest.uni_context_shape))

        final_states = flatten_final_states(encoder_output.final_states)
        true_final_states = flatten_final_states(build_state(RNNEncoderTest.hidden_state_shape, True))
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            final_states = sess.run(final_states)
            true_final_states = sess.run(true_final_states)
            self.assertAllEqual([x.shape for x in final_states],
                                [x.shape for x in true_final_states])

    def testStackUnidirectionalRNNEncoder2(self):
        encoder = rnn_encoder.UnidirectionalRNNEncoder(RNNEncoderTest.little_params2,
                                                       tf.contrib.learn.ModeKeys.TRAIN)
        encoder_output = encoder.encode(*build_inputs(*RNNEncoderTest.input_shape))
        self.assertAllEqual(encoder_output.outputs.shape,
                            tf.TensorShape(RNNEncoderTest.uni_context_shape))

        final_states = flatten_final_states(encoder_output.final_states)
        true_final_states = flatten_final_states(build_state(RNNEncoderTest.hidden_state_shape, True))
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            final_states = sess.run(final_states)
            true_final_states = sess.run(true_final_states)
            self.assertAllEqual([x.shape for x in final_states],
                                [x.shape for x in true_final_states])

    def testStackUnidirectionalRNNEncoder3(self):
        encoder = rnn_encoder.UnidirectionalRNNEncoder(RNNEncoderTest.little_params3,
                                                       tf.contrib.learn.ModeKeys.TRAIN)
        encoder_output = encoder.encode(*build_inputs(*RNNEncoderTest.input_shape))
        self.assertAllEqual(encoder_output.outputs.shape,
                            tf.TensorShape(RNNEncoderTest.uni_context_shape))

        final_states = flatten_final_states(encoder_output.final_states)
        true_final_states = flatten_final_states(build_state(RNNEncoderTest.hidden_state_shape, False))
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            final_states = sess.run(final_states)
            true_final_states = sess.run(true_final_states)
            self.assertAllEqual([x.shape for x in final_states],
                                [x.shape for x in true_final_states])

    def testStackBiUnidirectionalRNNEncoder1(self):
        encoder = rnn_encoder.BiUnidirectionalRNNEncoder(RNNEncoderTest.little_params1,
                                                         tf.contrib.learn.ModeKeys.TRAIN)
        encoder_output = encoder.encode(*build_inputs(*RNNEncoderTest.input_shape))
        self.assertAllEqual(encoder_output.outputs.shape,
                            tf.TensorShape(RNNEncoderTest.bi_context_shape))

        final_states = flatten_final_states(encoder_output.final_states)
        true_final_states = flatten_final_states({
            "forward": build_state(RNNEncoderTest.hidden_state_shape, True),
            "backward": build_state(RNNEncoderTest.hidden_state_shape, True)})
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            final_states = sess.run(final_states)
            true_final_states = sess.run(true_final_states)
            self.assertAllEqual([x.shape for x in final_states],
                                [x.shape for x in true_final_states])

    def testStackBiUnidirectionalRNNEncoder2(self):
        encoder = rnn_encoder.BiUnidirectionalRNNEncoder(RNNEncoderTest.little_params2,
                                                         tf.contrib.learn.ModeKeys.TRAIN)
        encoder_output = encoder.encode(*build_inputs(*RNNEncoderTest.input_shape))
        self.assertAllEqual(encoder_output.outputs.shape,
                            tf.TensorShape(RNNEncoderTest.uni_context_shape))

        final_states = flatten_final_states(encoder_output.final_states)
        true_final_states = build_state(RNNEncoderTest.hidden_state_shape, True)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            final_states = sess.run(final_states)
            true_final_states = sess.run(true_final_states)
            self.assertAllEqual([x.shape for x in final_states],
                                [x.shape for x in true_final_states])

    def testStackBiUnidirectionalRNNEncoder3(self):
        encoder = rnn_encoder.BiUnidirectionalRNNEncoder(RNNEncoderTest.little_params3,
                                                         tf.contrib.learn.ModeKeys.TRAIN)
        encoder_output = encoder.encode(*build_inputs(*RNNEncoderTest.input_shape))
        self.assertAllEqual(encoder_output.outputs.shape,
                            tf.TensorShape(RNNEncoderTest.uni_context_shape))

        final_states = flatten_final_states(encoder_output.final_states)
        true_final_states = flatten_final_states(build_state(RNNEncoderTest.hidden_state_shape, False))
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            final_states = sess.run(final_states)
            true_final_states = sess.run(true_final_states)
            self.assertAllEqual([x.shape for x in final_states],
                                [x.shape for x in true_final_states])

    def testStackBidirectionalRNNEncoder1(self):
        encoder = rnn_encoder.StackBidirectionalRNNEncoder(RNNEncoderTest.little_params1,
                                                           tf.contrib.learn.ModeKeys.TRAIN)
        encoder_output = encoder.encode(*build_inputs(*RNNEncoderTest.input_shape))
        self.assertAllEqual(encoder_output.outputs.shape,
                            tf.TensorShape(RNNEncoderTest.bi_context_shape))

        final_states = flatten_final_states(encoder_output.final_states)
        true_final_states = flatten_final_states({
            "forward": build_state(RNNEncoderTest.hidden_state_shape, True),
            "backward": build_state(RNNEncoderTest.hidden_state_shape, True)})
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            final_state = sess.run(final_states)
            true_final_state = sess.run(true_final_states)
            self.assertAllEqual([x.shape for x in final_state],
                                [x.shape for x in true_final_state])

    def testStackBidirectionalRNNEncoder2(self):
        encoder = rnn_encoder.StackBidirectionalRNNEncoder(RNNEncoderTest.little_params2,
                                                           tf.contrib.learn.ModeKeys.TRAIN)
        encoder_output = encoder.encode(*build_inputs(*RNNEncoderTest.input_shape))
        self.assertAllEqual(encoder_output.outputs.shape,
                            tf.TensorShape(RNNEncoderTest.bi_context_shape))

        final_states = flatten_final_states(encoder_output.final_states)
        true_final_states = flatten_final_states({
            "forward": build_state(RNNEncoderTest.hidden_state_shape, True),
            "backward": build_state(RNNEncoderTest.hidden_state_shape, True)})
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            final_state = sess.run(final_states)
            true_final_state = sess.run(true_final_states)
            self.assertAllEqual([x.shape for x in final_state],
                                [x.shape for x in true_final_state])

    def testStackBidirectionalRNNEncoder3(self):
        encoder = rnn_encoder.StackBidirectionalRNNEncoder(RNNEncoderTest.little_params3,
                                                           tf.contrib.learn.ModeKeys.TRAIN)
        encoder_output = encoder.encode(*build_inputs(*RNNEncoderTest.input_shape))
        self.assertAllEqual(encoder_output.outputs.shape,
                            tf.TensorShape(RNNEncoderTest.bi_context_shape))

        final_states = flatten_final_states(encoder_output.final_states)
        true_final_states = flatten_final_states({
            "forward": build_state(RNNEncoderTest.hidden_state_shape, False),
            "backward": build_state(RNNEncoderTest.hidden_state_shape, False)})
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            final_state = sess.run(final_states)
            true_final_state = sess.run(true_final_states)
            self.assertAllEqual([x.shape for x in final_state],
                                [x.shape for x in true_final_state])


if __name__ == "__main__":
    tf.test.main()
