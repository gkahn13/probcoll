from general.tf import tf_utils
import tensorflow as tf

def separable_convnn(
        inputs,
        params,
        scope='convnn',
        dtype=tf.float32,
        data_format='NHWC',
        reuse=False,
        is_training=True):

    if params['conv_activation'] == 'relu':
        conv_activation = tf.nn.relu
    else:
        raise NotImplementedError(
            'Conv activation {0} is not valid'.format(
                params['conv_activation']))

    if 'output_activation' not in params:
        output_activation = None
    elif params['output_activation'] == 'sigmoid':
        output_activation = tf.nn.sigmoid
    elif params['output_activation'] == 'softmax':
        output_activation = tf.nn.softmax
    elif params['output_activation'] == 'spatial_softmax':
        output_activation = lambda x: tf_utils.spatial_soft_argmax(x, dtype) 
    elif params['output_activation'] == 'tanh':
        output_activation = tf.nn.tanh
    elif params['output_activation'] == 'relu':
        output_activation = tf.nn.relu
    else:
        raise NotImplementedError(
            'Output activation {0} is not valid'.format(
                params['output_activation']))
    
    kernels = params['kernels']
    filters = params['filters']
    strides = params['strides']
    # Assuming all paddings will be the same type
    padding = params['padding']

    assert(data_format == 'NHWC')
    next_layer_input = inputs
    with tf.variable_scope(scope, reuse=reuse):
        for i in range(len(kernels)):
            if i == len(kernels) - 1:
                activation = output_activation
            else:
                activation = conv_activation
            if params.get('use_batch_norm', False):
                normalizer_fn = tf.contrib.layers.batch_norm
                scale = not (activation == tf.nn.relu or activation is None) 
                normalizer_params = {
                        'is_training': is_training,
                        'data_format': data_format,
                        'fused': True,
                        'decay': params.get('batch_norm_decay', 0.999),
                        'zero_debias_moving_mean': True,
                        'scale': scale
                    }
            else:
                normalizer_fn = None
                normalizer_params = None
            next_layer_input = tf.contrib.layers.separable_conv2d(
                inputs=next_layer_input,
                num_outputs=filters[i],
                kernel_size=kernels[i],
                depth_multiplier=1, # TODO
                stride=strides[i],
                padding=padding,
                activation_fn=activation,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params,
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype),
                weights_regularizer=tf.contrib.layers.l2_regularizer(0.5),
                biases_initializer=tf.constant_initializer(0., dtype=dtype),
                trainable=True)

    output = next_layer_input
    # TODO
    return output, None