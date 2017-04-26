import tensorflow as tf
from general.tf import tf_utils

class BasicMulintRNNCell(tf.nn.rnn_cell.BasicRNNCell):
    def __call__(self, inputs, state, scope=None):
        """Most basic RNN: output = new_state = tanh(W * input + U * state + B)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
            output = self._activation(
		tf_utils.multiplicative_integration(
			[inputs, state], self._num_units))
        return output, output 
