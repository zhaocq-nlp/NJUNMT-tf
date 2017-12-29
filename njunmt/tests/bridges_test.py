import numpy
import tensorflow as tf

from njunmt.encoders import rnn_encoder
from njunmt.utils import bridges
from njunmt.utils.rnn_cell_utils import get_multilayer_rnn_cells


def build_inputs(batch_size, max_len, dim):
    x = numpy.random.random(size=(batch_size, max_len, dim))
    x_len = numpy.random.randint(low=1, high=max_len, size=(batch_size,))
    return tf.convert_to_tensor(x, dtype=tf.float32), \
           tf.convert_to_tensor(x_len, dtype=tf.int32)


def build_state(shape, is_lstm):
    state = tf.convert_to_tensor(numpy.zeros(shape=shape), dtype=tf.float32)
    if is_lstm:
        return tf.contrib.rnn.LSTMStateTuple(state, state)
    return state


class BridgeTest(tf.test.TestCase):
    batch_size = 7
    max_len = 11
    dim = 17
    hidden_dim = 5
    beam_size = 3

    input_shape = (batch_size, max_len, dim)
    hidden_state_shape = (batch_size, hidden_dim)
    hidden_state_shape_wb = (batch_size * beam_size, hidden_dim)
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

    state_size1 = get_multilayer_rnn_cells(**little_params1["rnn_cell"]).state_size
    state_size2 = get_multilayer_rnn_cells(**little_params2["rnn_cell"]).state_size
    state_size3 = get_multilayer_rnn_cells(**little_params3["rnn_cell"]).state_size

    def testPassThroughBridge(self):
        # TODO
        # encoder1 = rnn_encoder.StackBidirectionalRNNEncoder(BridgeTest.little_params1, tf.contrib.learn.ModeKeys.TRAIN)
        encoder2 = rnn_encoder.StackBidirectionalRNNEncoder(BridgeTest.little_params3, tf.contrib.learn.ModeKeys.TRAIN)
        encoder3 = rnn_encoder.UnidirectionalRNNEncoder(BridgeTest.little_params2, tf.contrib.learn.ModeKeys.TRAIN)
        # encoder_output1 = encoder1.encode(*build_inputs(*BridgeTest.input_shape), scope="1")
        encoder_output2 = encoder2.encode(*build_inputs(*BridgeTest.input_shape), scope="2")
        encoder_output3 = encoder3.encode(*build_inputs(*BridgeTest.input_shape), scope="3")
        # bridge1 = bridges.PassThroughBridge({}, encoder_output1, tf.contrib.learn.ModeKeys.TRAIN)
        # bridge4 = bridges.PassThroughBridge({"direction": "forward"}, encoder_output1, tf.contrib.learn.ModeKeys.TRAIN)
        bridge2 = bridges.PassThroughBridge({"direction": "backward"}, encoder_output2, tf.contrib.learn.ModeKeys.TRAIN)
        bridge3 = bridges.PassThroughBridge({"direction": "forward"}, encoder_output3, tf.contrib.learn.ModeKeys.TRAIN)
        # state4 = bridge4(BridgeTest.state_size1)
        # true_state4 = (encoder_output1.final_states["forward"],)
        state2 = bridge2(BridgeTest.state_size2)
        true_state2 = (encoder_output2.final_states["backward"],
                       encoder_output2.final_states["backward"],
                       encoder_output2.final_states["backward"],)
        state3 = bridge3(BridgeTest.state_size3)
        true_state3 = (encoder_output3.final_states,
                       encoder_output3.final_states,
                       encoder_output3.final_states,)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllEqual(sess.run(state2), sess.run(true_state2))
            self.assertAllEqual(sess.run(state3), sess.run(true_state3))

    def testInitialStateBridge(self):
        # TODO
        pass

    def testVariableBridge(self):
        encoder1 = rnn_encoder.StackBidirectionalRNNEncoder(BridgeTest.little_params1, tf.contrib.learn.ModeKeys.TRAIN)
        encoder2 = rnn_encoder.StackBidirectionalRNNEncoder(BridgeTest.little_params2, tf.contrib.learn.ModeKeys.TRAIN)
        encoder3 = rnn_encoder.StackBidirectionalRNNEncoder(BridgeTest.little_params3, tf.contrib.learn.ModeKeys.TRAIN)
        encoder_output1 = encoder1.encode(*build_inputs(*BridgeTest.input_shape), scope="1")
        encoder_output2 = encoder2.encode(*build_inputs(*BridgeTest.input_shape), scope="2")
        encoder_output3 = encoder3.encode(*build_inputs(*BridgeTest.input_shape), scope="3")
        bridge1 = bridges.VariableBridge({}, encoder_output1, tf.contrib.learn.ModeKeys.TRAIN)
        bridge2 = bridges.VariableBridge({}, encoder_output2, tf.contrib.learn.ModeKeys.TRAIN)
        bridge3 = bridges.VariableBridge({}, encoder_output3, tf.contrib.learn.ModeKeys.TRAIN)
        state1 = bridge1(BridgeTest.state_size1, name="1")
        true_state1 = (build_state(BridgeTest.hidden_state_shape, True),)
        state2 = bridge2(BridgeTest.state_size2, name="2")
        true_state2 = (
            build_state(BridgeTest.hidden_state_shape, True), build_state(BridgeTest.hidden_state_shape, True),
            build_state(BridgeTest.hidden_state_shape, True),)
        state3 = bridge3(BridgeTest.state_size3, name="3")
        true_state3 = (
            build_state(BridgeTest.hidden_state_shape, False), build_state(BridgeTest.hidden_state_shape, False),
            build_state(BridgeTest.hidden_state_shape, False),)
        state1b = bridge1(BridgeTest.state_size1, beam_size=BridgeTest.beam_size, name="4")
        true_state1b = (build_state(BridgeTest.hidden_state_shape_wb, True),)
        state2b = bridge2(BridgeTest.state_size2, beam_size=BridgeTest.beam_size, name="5")
        true_state2b = (
            build_state(BridgeTest.hidden_state_shape_wb, True), build_state(BridgeTest.hidden_state_shape_wb, True),
            build_state(BridgeTest.hidden_state_shape_wb, True),)
        state3b = bridge3(BridgeTest.state_size3, beam_size=BridgeTest.beam_size, name="6")
        true_state3b = (
            build_state(BridgeTest.hidden_state_shape_wb, False), build_state(BridgeTest.hidden_state_shape_wb, False),
            build_state(BridgeTest.hidden_state_shape_wb, False),)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllEqual(sess.run(state1), sess.run(true_state1))
            self.assertAllEqual(sess.run(state2), sess.run(true_state2))
            self.assertAllEqual(sess.run(state3), sess.run(true_state3))
            self.assertAllEqual(sess.run(state1b), sess.run(true_state1b))
            self.assertAllEqual(sess.run(state2b), sess.run(true_state2b))
            self.assertAllEqual(sess.run(state3b), sess.run(true_state3b))

    def testZeroBridge(self):
        encoder1 = rnn_encoder.StackBidirectionalRNNEncoder(BridgeTest.little_params1, tf.contrib.learn.ModeKeys.TRAIN)
        encoder2 = rnn_encoder.StackBidirectionalRNNEncoder(BridgeTest.little_params2, tf.contrib.learn.ModeKeys.TRAIN)
        encoder3 = rnn_encoder.StackBidirectionalRNNEncoder(BridgeTest.little_params3, tf.contrib.learn.ModeKeys.TRAIN)
        encoder_output1 = encoder1.encode(*build_inputs(*BridgeTest.input_shape), scope="1")
        encoder_output2 = encoder2.encode(*build_inputs(*BridgeTest.input_shape), scope="2")
        encoder_output3 = encoder3.encode(*build_inputs(*BridgeTest.input_shape), scope="3")
        bridge1 = bridges.ZeroBridge({}, encoder_output1, tf.contrib.learn.ModeKeys.TRAIN)
        bridge2 = bridges.ZeroBridge({}, encoder_output2, tf.contrib.learn.ModeKeys.TRAIN)
        bridge3 = bridges.ZeroBridge({}, encoder_output3, tf.contrib.learn.ModeKeys.TRAIN)

        state1 = bridge1(BridgeTest.state_size1)
        true_state1 = (build_state(BridgeTest.hidden_state_shape, True),)
        state2 = bridge2(BridgeTest.state_size2)
        true_state2 = (
            build_state(BridgeTest.hidden_state_shape, True), build_state(BridgeTest.hidden_state_shape, True),
            build_state(BridgeTest.hidden_state_shape, True),)
        state3 = bridge3(BridgeTest.state_size3)
        true_state3 = (
            build_state(BridgeTest.hidden_state_shape, False), build_state(BridgeTest.hidden_state_shape, False),
            build_state(BridgeTest.hidden_state_shape, False),)
        state1b = bridge1(BridgeTest.state_size1, beam_size=BridgeTest.beam_size)
        true_state1b = (build_state(BridgeTest.hidden_state_shape_wb, True),)
        state2b = bridge2(BridgeTest.state_size2, beam_size=BridgeTest.beam_size)
        true_state2b = (
            build_state(BridgeTest.hidden_state_shape_wb, True), build_state(BridgeTest.hidden_state_shape_wb, True),
            build_state(BridgeTest.hidden_state_shape_wb, True),)
        state3b = bridge3(BridgeTest.state_size3, beam_size=BridgeTest.beam_size)
        true_state3b = (
            build_state(BridgeTest.hidden_state_shape_wb, False), build_state(BridgeTest.hidden_state_shape_wb, False),
            build_state(BridgeTest.hidden_state_shape_wb, False),)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllEqual(sess.run(state1), sess.run(true_state1))
            self.assertAllEqual(sess.run(state2), sess.run(true_state2))
            self.assertAllEqual(sess.run(state3), sess.run(true_state3))
            self.assertAllEqual(sess.run(state1b), sess.run(true_state1b))
            self.assertAllEqual(sess.run(state2b), sess.run(true_state2b))
            self.assertAllEqual(sess.run(state3b), sess.run(true_state3b))


if __name__ == "__main__":
    tf.test.main()
