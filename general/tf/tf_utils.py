import tensorflow as tf

# TODO just summing might be better
def cumulative_increasing_sum(x, dtype=tf.float32):
    """
    x is tensor of shape (batch_size x T).
    output[i] = sum(x[:i+1])
    return output
    """
    # TODO 
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

def linear(args, output_size, dtype=tf.float32, scope=None):
    with tf.variable_scope(scope or "linear"):
        if isinstance(args, list) or isinstance(args, tuple):
            if len(args) != 1:
                inputs = tf.concat(1, args)
            else:
                inputs = args[0]
        else:
            inputs = args
            args = [args]
        total_arg_size = 0
        shapes = [a.get_shape() for a in args]
        for shape in shapes:
            if shape.ndims != 2:
                raise ValueError("linear is expecting 2D arguments: %s" % shapes)
            else:
                total_arg_size += shape[1].value
        dtype = args[0].dtype
        weights = tf.get_variable(
            "weights",
            [total_arg_size, output_size],
            dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer(dtype=dtype))
        output = tf.matmul(inputs, weights)
    return output

def multiplicative_integration(
            list_of_inputs,
            output_size,
            initial_bias_value=0.0,
            weights_already_calculated=False,
            reg_collection=None,
            dtype=tf.float32,
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
            Wx = linear(
                list_of_inputs[0],
                output_size,
                dtype=dtype,
                reg_collection=reg_collection,
                scope="Calculate_Wx_mulint")

            Uz = linear(
                list_of_inputs[1],
                output_size,
                dtype=dtype,
                reg_collection=reg_collection,
                scope="Calculate_Uz_mulint")

        with tf.variable_scope("multiplicative_integration"):
            alpha = tf.get_variable(
                'mulint_alpha',
                [output_size],
                dtype=dtype,
                initializer=tf.truncated_normal_initializer(
                    mean=1.0,
                    stddev=0.1,
                    dtype=dtype))

            beta1, beta2 = tf.split(0, 2, tf.get_variable(
                'mulint_params_betas',
                [output_size * 2],
                dtype=dtype,
                initializer=tf.truncated_normal_initializer(
                    mean=0.5,
                    stddev=0.1,
                    dtype=dtype)))

            original_bias = tf.get_variable(
                'mulint_original_bias',
                [output_size],
                dtype=dtype,
                initializer=tf.truncated_normal_initializer(
                    mean=initial_bias_value,
                    stddev=0.1,
                    dtype=dtype))

        final_output = alpha * Wx * Uz + beta1 * Uz + beta2 * Wx + original_bias

    return final_output

def spatial_soft_argmax(features, dtype=tf.float32):
    """
    features shape is [N, H, W, C]
    """
    shape = tf.shape(features)
    N, H, W, C = shape[0], shape[1], shape[2], shape[3]
    features = tf.reshape(
        tf.transpose(features, [0, 3, 1, 2]),
        [-1, H * W])
    softmax = tf.nn.softmax(features)
    spatial_softmax = tf.transpose(tf.reshape(softmax, [N, C, H, W]), [0, 2, 3, 1])
    spatial_softmax_pos = tf.expand_dims(spatial_softmax, -1)
    # TODO shape [H, W, 1, 2]
    # TODO H or W is 1
    delta_h = 2. / tf.cast(H - 1, dtype)
    delta_w = 2. / tf.cast(W - 1, dtype)
    ran_h = tf.tile(tf.expand_dims(tf.range(-1., 1. + delta_h, delta_h, dtype=dtype), 1), [1, W])
    ran_w = tf.tile(tf.expand_dims(tf.range(-1., 1 + delta_w, delta_w, dtype=dtype), 0), [H, 1])
    image_pos = tf.expand_dims(tf.stack([ran_h, ran_w], 2), 2)
    spatial_soft_amax = tf.reduce_sum(spatial_softmax_pos * image_pos, axis=[1, 2])
    return tf.reshape(spatial_soft_amax, [N, C * 2])

def str_to_dtype(dtype):
    if dtype == "float32":
        return tf.float32
    elif dtype == "float16":
        return tf.float16
    elif dtype == "float64":
        return tf.float64
    elif dtype == "int32":
        return tf.int32
    elif dtype == "uint8":
        return tf.uint8
    else:
        raise NotImplementedError(
            "dtype {0} is not valid".format(dtype))
