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
            
            ins = tf.concat([inputs, state], axis=1)
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
            use_layer_norm=False,
            weights_scope=None):
        
        self._num_units = num_units
        self._dropout_mask = dropout_mask
        self._activation = activation
        self._dtype = dtype
        self._use_layer_norm = use_layer_norm
        
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
            if self._use_layer_norm:
                Wx = tf.contrib.layers.layer_norm(
                    Wx,
                    center=False,
                    scale=False)
                Uz = tf.contrib.layers.layer_norm(
                    Uz,
                    center=False,
                    scale=False)
            output = self._activation(
		tf_utils.multiplicative_integration(
                    [Wx, Uz],
                    self._num_units,
                    dtype=self._dtype,
                    weights_already_calculated=True))
            
            if self._dropout_mask is not None:
                output = output * self._dropout_mask
        
        return output, output

class DpLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    
    def __init__(
            self,
            num_units,
            forget_bias=1.0,
            dropout_mask=None,
            activation=tf.tanh,
            dtype=tf.float32,
            num_inputs=None,
            weights_scope=None):
        
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._dropout_mask = dropout_mask
        self._activation = activation
        self._dtype = dtype
        self._state_is_tuple = True

        with tf.variable_scope(weights_scope or type(self).__name__):
            self._weights = tf.get_variable(
                "weights",
                [num_inputs + num_units, 4 * num_units],
                dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
                regularizer=tf.contrib.layers.l2_regularizer(0.5))

    def __call__(
            self,
            inputs,
            state,
            scope=None):
        """Most basic LSTM with same dropout at every time step."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
            
            c, h = state
            ins = tf.concat([inputs, h], axis=1)
            output = self._activation(tf.matmul(ins, self._weights))

            i, j, f, o = tf.split(output, 4, axis=1)
            
            forget = c * tf.nn.sigmoid(f + self._forget_bias)
            new = tf.nn.sigmoid(i) * self._activation(j)
            new_c = forget + new
            
            # TODO make sure this is correct
            if self._dropout_mask is not None:
                new_c = new_c * self._dropout_mask

            new_h = self._activation(new_c) * tf.nn.sigmoid(o)
            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

        return new_h, new_state

class DpMulintLSTMCell(DpLSTMCell):
    
    def __init__(
            self,
            num_units,
            forget_bias=1.0,
            dropout_mask=None,
            activation=tf.tanh,
            dtype=tf.float32,
            num_inputs=None,
            use_layer_norm=False,
            weights_scope=None):
        
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._dropout_mask = dropout_mask
        self._activation = activation
        self._dtype = dtype
        self._use_layer_norm = use_layer_norm
        self._state_is_tuple = True
        
        with tf.variable_scope(weights_scope or type(self).__name__):
            self._weights_W = tf.get_variable(
                "weights_W",
                [num_inputs, 4 * num_units],
                dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
                regularizer=tf.contrib.layers.l2_regularizer(0.5))

            self._weights_U = tf.get_variable(
                "weights_U",
                [num_units, 4 * num_units],
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
            
            c, h = state
            
            Wx = tf.matmul(inputs, self._weights_W)
            Uz = tf.matmul(h, self._weights_U)
            if self._use_layer_norm:
                Wx = tf.contrib.layers.layer_norm(
                    Wx,
                    center=False,
                    scale=False)
                Uz = tf.contrib.layers.layer_norm(
                    Uz,
                    center=False,
                    scale=False)
            output = self._activation(
		tf_utils.multiplicative_integration(
                    [Wx, Uz],
                    4 * self._num_units,
                    dtype=self._dtype,
                    weights_already_calculated=True))
            
            i, j, f, o = tf.split(output, 4, axis=1)
            
            forget = c * tf.nn.sigmoid(f + self._forget_bias)
            new = tf.nn.sigmoid(i) * self._activation(j)
            new_c = forget + new

#            if self._use_layer_norm:
#                new_c = tf.contrib.layers.layer_norm(
#                    new_c,
#                    center=True,
#                    scale=True)

            # TODO make sure this is correct
            if self._dropout_mask is not None:
                new_c = new_c * self._dropout_mask

            if self._use_layer_norm:
                norm_c = tf.contrib.layers.layer_norm(
                    new_c,
                    center=True,
                    scale=True)
                new_h = self._activation(norm_c) * tf.nn.sigmoid(o)
            else:
                new_h = self._activation(new_c) * tf.nn.sigmoid(o)
            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
        return new_h, new_state
