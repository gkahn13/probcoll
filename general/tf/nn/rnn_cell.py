import tensorflow as tf
from general.tf import tf_utils

class DpRNNCell(tf.nn.rnn_cell.BasicRNNCell):
    
    def __init__(
            self,
            num_units,
            dropout_mask=None,
            activation=tf.tanh,
            dtype=tf.float32,
            num_inputs=None,
            weights_scope=None):
        
        self._num_units = num_units
        self._dropout_mask = dropout_mask
        self._activation = activation
        self._dtype = dtype
        
        with tf.variable_scope(weights_scope or type(self).__name__):
            self._weights = tf.get_variable(
                "weights",
                [num_inputs + num_units, num_units],
                dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
                regularizer=tf.contrib.layers.l2_regularizer(0.5))

    def __call__(
            self,
            inputs,
            state,
            scope=None):
        """Most basic RNN: output = new_state = tanh(W * input + U * state + B). With same dropout at every time step."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
            
            ins = tf.concat(1, [inputs, state])
            output = self._activation(tf.matmul(ins, self._weights))

            if self._dropout_mask is not None:
                output = output * self._dropout_mask

        return output, output

class DpMulintRNNCell(DpRNNCell):
    
    def __init__(
            self,
            num_units,
            dropout_mask=None,
            activation=tf.tanh,
            dtype=tf.float32,
            num_inputs=None,
            weights_scope=None):
        
        self._num_units = num_units
        self._dropout_mask = dropout_mask
        self._activation = activation
        self._dtype = dtype
        
        with tf.variable_scope(weights_scope or type(self).__name__):
            self._weights_W = tf.get_variable(
                "weights_W",
                [num_inputs, num_units],
                dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
                regularizer=tf.contrib.layers.l2_regularizer(0.5))

            self._weights_U = tf.get_variable(
                "weights_U",
                [num_units, num_units],
                dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
                regularizer=tf.contrib.layers.l2_regularizer(0.5))

    def __call__(
            self,
            inputs,
            state,
            scope=None):
        """Most basic RNN: output = new_state = tanh(W * input + U * state + B)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
            Wx = tf.matmul(inputs, self._weights_W)
            Uz = tf.matmul(state, self._weights_U)
            output = self._activation(
		tf_utils.multiplicative_integration(
                    [Wx, Uz],
                    self._num_units,
                    dtype=self._dtype,
                    weights_already_calculated=True))
            
            if self._dropout_mask is not None:
                output = output * self._dropout_mask
        
        return output, output

