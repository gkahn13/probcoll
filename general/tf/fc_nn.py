import tensorflow as tf

def fcnn(
        inputs,
        params,
        use_dp_placeholders=False,
        scope="fcnn",
        reuse=False):
    
    if params["hidden_activation"] == "relu":
        hidden_activation = tf.nn.relu
    elif params["hidden_activation"] == "tanh":
        hidden_activation = tf.nn.tanh
    elif params["hidden_activation"] == "None":
        hidden_activation = None
    else:
        raise NotImplementedError(
            "Hidden activation {0} is not valid".format(
                params["hidden_activation"]))

    if params["output_activation"] == "sigmoid":
        output_activation = tf.nn.sigmoid
    elif params["output_activation"] == "softmax":
        output_activation = tf.nn.softmax
    elif params["output_activation"] == "None":
        output_activation = None
    else:
        raise NotImplementedError(
            "Output activation {0} is not valid".format(
                params["output_activation"]))

    hidden_layers = params["hidden_layers"]
    output_dim = params["output_dim"]
    # TODO
    dtype = tf.float32
#    dtype = params["dtype"]
    dropout = params["dropout"]
    if use_dp_placeholders:
        dropout_placeholders = []
    else:
        dropout_placeholders = None

    next_layer_input = inputs
    with tf.variable_scope(scope, reuse=reuse):
        for dim in hidden_layers:
            next_layer_input = tf.contrib.layers.fully_connected(
                inputs=next_layer_input,
                num_outputs=dim,
                activation_fn=hidden_activation,
                weights_initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
                biases_initializer=tf.constant_initializer(0., dtype=dtype),
                weights_regularizer=tf.contrib.layers.l2_regularizer(0.5),
                trainable=True)

            if dropout is not None:
                assert(type(dropout) is float and 0 <= dropout and dropout <= 1.0)
                if use_dp_placeholders:
                    dp = tf.placeholder(tf.float32, [None, dim])
                    dropout_placeholders.append(dp)
                    next_layer_input = next_layer_input * dp
                else:
                    next_layer_input = tf.nn.dropout(next_layer_input, dropout)


        with tf.name_scope('scope_output_pred'):
            output = tf.contrib.layers.fully_connected(
                inputs=next_layer_input,
                num_outputs=output_dim,
                activation_fn=output_activation,
                weights_initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
                biases_initializer=tf.constant_initializer(0., dtype=dtype),
                weights_regularizer=tf.contrib.layers.l2_regularizer(0.5),
                trainable=True)

    return output, dropout_placeholders
