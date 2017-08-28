from general.tf import tf_utils
import tensorflow as tf

def convnn(
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
    normalizer = params.get('normalizer', None)
    next_layer_input = inputs
    with tf.variable_scope(scope, reuse=reuse):
        for i in range(len(kernels)):
            if i == len(kernels) - 1:
                activation = output_activation
            else:
                activation = conv_activation
            if normalizer == 'batch_norm':
                normalizer_fn = tf.contrib.layers.batch_norm
                normalizer_params = {
                        'is_training': is_training,
                        'data_format': data_format,
                        'fused': True,
                        'decay': params.get('batch_norm_decay', 0.999),
                        'zero_debias_moving_mean': True,
                        'scale': True,
                        'center': True,
                        'updates_collections': None
                    }
            elif normalizer == 'layer_norm':
                normalizer_fn = tf.contrib.layers.layer_norm
                normalizer_params = {
                        'scale': True,
                        'center': True
                    }
            elif normalizer is None:
                normalizer_fn = None
                normalizer_params = None
            else:
                raise NotImplementedError(
                    'Normalizer {0} is not valid'.format(normalizer))
            next_layer_input = tf.contrib.layers.conv2d(
                inputs=next_layer_input,
                num_outputs=filters[i],
                data_format=data_format,
                kernel_size=kernels[i],
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
