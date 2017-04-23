import tensorflow as tf

# TODO just summing might be better
def cumulative_increasing_sum(x):
    """
    x is tensor of shape (batch_size x T).
    output[i] = sum(x[:i+1])
    return output
    """
    # TODO 
    dtype = tf.float32
    T = tf.shape(x)[1]
    mask1 = tf.concat(0, [
            tf.ones((1,), dtype=dtype), tf.zeros((T - 1,), dtype=dtype)
        ])
    mask2 = tf.concat(0, [
            tf.zeros((1,), dtype=dtype), tf.ones((T - 1,), dtype=dtype)
        ])
    masked = x * mask1 + tf.nn.relu(x) * mask2
    upper_triangle = tf.matrix_band_part(
        tf.ones((T, T), dtype=dtype),
        0,
        -1)
    output = tf.matmul(masked, upper_triangle)
    return output

def multiplicative_integration(
            list_of_inputs,
            output_size,
            initial_bias_value=0.0,
            weights_already_calculated=False,
            use_l2_loss=False,
            scope=None):
    '''
    expects len(2) for list of inputs and will perform integrative multiplication
    weights_already_calculated will treat the list of inputs as Wx and Uz and is useful for batch normed inputs
    '''
    with tf.variable_scope(scope or 'double_inputs_multiple_integration'):
        if len(list_of_inputs) != 2: raise ValueError('list of inputs must be 2, you have:', len(list_of_inputs))

        # TODO can do batch norm in FC
        if weights_already_calculated:  # if you already have weights you want to insert from batch norm
            Wx = list_of_inputs[0]
            Uz = list_of_inputs[1]

        else:
            # TODO get regularizer to work
            if use_l2_loss:
                regularizer = None
            else:
                regularizer = tf.contrib.layers.l2_regularizer(0.5)
            with tf.variable_scope('Calculate_Wx_mulint'):
                Wx = tf.contrib.layers.fully_connected(
                    inputs=list_of_inputs[0],
                    num_outputs=output_size,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
#                    weights_regularizer=regularizer,
                    trainable=True)

            with tf.variable_scope("Calculate_Uz_mulint"):
                Uz = tf.contrib.layers.fully_connected(
                    inputs=list_of_inputs[1],
                    num_outputs=output_size,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
#                    weights_regularizer=regularizer,
                    trainable=True)

        with tf.variable_scope("multiplicative_integration"):
            alpha = tf.get_variable(
                'mulint_alpha',
                [output_size],
                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.1))

            beta1, beta2 = tf.split(0, 2, tf.get_variable(
                'mulint_params_betas',
                [output_size * 2],
                initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.1)))

            original_bias = tf.get_variable(
                'mulint_original_bias',
                [output_size],
                initializer=tf.truncated_normal_initializer(
                    mean=initial_bias_value,
                    stddev=0.1))

        final_output = alpha * Wx * Uz + beta1 * Uz + beta2 * Wx + original_bias

    return final_output
