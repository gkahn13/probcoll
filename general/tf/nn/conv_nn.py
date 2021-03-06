from general.tf import tf_utils
import tensorflow as tf

def convnn(inputs, params, scope="convnn", dtype=tf.float32, reuse=False, is_training=True):

    if params["conv_activation"] == "relu":
        conv_activation = tf.nn.relu
    else:
        raise NotImplementedError(
            "Conv activation {0} is not valid".format(
                params["conv_activation"]))

    if "output_activation" not in params:
        output_activation = None
    elif params["output_activation"] == "sigmoid":
        output_activation = tf.nn.sigmoid
    elif params["output_activation"] == "softmax":
        output_activation = tf.nn.softmax
    elif params['output_activation'] == 'spatial_softmax':
        output_activation = lambda x: tf_utils.spatial_soft_argmax(x, dtype) 
    elif params["output_activation"] == "tanh":
        output_activation = tf.nn.tanh
    elif params['output_activation'] == 'relu':
        output_activation = tf.nn.relu
    else:
        raise NotImplementedError(
            "Output activation {0} is not valid".format(
                params["output_activation"]))
    
    kernels = params["kernels"]
    filters = params["filters"]
    strides = params["strides"]
    # Assuming all paddings will be the same type
    padding = params["padding"]
    next_layer_input = inputs
    if params.get('use_batch_norm', False):
        normalizer_fn = tf.contrib.layers.batch_norm
        normalizer_params = {
                'is_training': is_training,
            }
    else:
        normalizer_fn = None
        normalizer_params = None
    with tf.variable_scope(scope, reuse=reuse):
        for i in xrange(len(kernels)):
            next_layer_input = tf.contrib.layers.conv2d(
                inputs=next_layer_input,
                num_outputs=filters[i],
                kernel_size=kernels[i],
                stride=strides[i],
                padding=padding,
                activation_fn=conv_activation,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params,
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype),
                weights_regularizer=tf.contrib.layers.l2_regularizer(0.5),
                biases_initializer=tf.constant_initializer(0., dtype=dtype),
                trainable=True)

    output = next_layer_input
    # TODO
    return output, None
