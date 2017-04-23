import tensorflow as tf

def convnn(inputs, params, scope="convnn", reuse=False):

    if params["conv_activation"] == "relu":
        conv_activation = tf.nn.relu
    else:
        raise NotImplementedError(
            "Conv activation {0} is not valid".format(
                params["conv_activation"]))

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
    
    kernels = params["kernels"]
    filters = params["filters"]
    strides = params["strides"]
    # Assuming all paddings will be the same type
    padding = params["padding"]
    # TODO
    dtype = tf.float32
    next_layer_input = inputs
    with tf.variable_scope(scope, reuse=reuse):
        for i in xrange(len(kernels)):
            next_layer_input = tf.contrib.layers.conv2d(
                inputs=next_layer_input,
                num_outputs=filters[i],
                kernel_size=kernels[i],
                stride=strides[i],
                padding=padding,
                activation_fn=conv_activation,
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype),
                weights_regularizer=tf.contrib.layers.l2_regularizer(0.5),
                biases_initializer=tf.constant_initializer(0., dtype=dtype),
                trainable=True)

    output = next_layer_input
    return output
