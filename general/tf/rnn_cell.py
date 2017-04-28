import tensorflow as tf
from general.tf import tf_utils

class DpRNNCell(tf.nn.rnn_cell.BasicRNNCell):
    
    def __init__(self, num_units, dropout_mask=None, activation=tf.tanh):
        self._num_units = num_units
        self._dropout_mask = dropout_mask
        self._activation = activation
    
    def __call__(
            self,
            inputs,
            state,
            scope=None):
        """Most basic RNN: output = new_state = tanh(W * input + U * state + B). With same dropout at every time step."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
            
            inputs = tf.concat(1, inputs, state)
            outputs = tf.contrib.layers.fully_connected(
                inputs=inputs,
                num_ouptuts=self._num_units,
                weights_initializer=tf.contrib.layers.xavier_intializer(),
                scope="linear",
                trainable=True)

            if self._dropout_mask is not None:
                output = output * self._dropout_mask

        return output, output

class DpMulintRNNCell(DpRNNCell):
    
    def __call__(
            self,
            inputs,
            state,
            scope=None):
        """Most basic RNN: output = new_state = tanh(W * input + U * state + B)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
            output = self._activation(
		tf_utils.multiplicative_integration(
			[inputs, state], self._num_units))
            
            if self._dropout_mask is not None:
                output = output * self._dropout_mask
        
        return output, output

