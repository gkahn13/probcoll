from general.tf import tf_utils
import tensorflow as tf

def fcnn(
        inputs,
        params,
        dp_masks=None,
        num_dp=1,
        dtype=tf.float32,
        data_format='NCHW',
        scope='fcnn',
        reuse=False,
        is_training=True,
        T=None):
    
    if 'hidden_activation' not in params:
        hidden_activation = None
    elif params['hidden_activation'] == 'relu':
        hidden_activation = tf.nn.relu
    elif params['hidden_activation'] == 'tanh':
        hidden_activation = tf.nn.tanh
    else:
        raise NotImplementedError(
            'Hidden activation {0} is not valid'.format(
                params['hidden_activation']))

    if 'output_activation' not in params or params['output_activation'] == 'None':
        output_activation = None
    elif params['output_activation'] == 'sigmoid':
        output_activation = tf.nn.sigmoid
    elif params['output_activation'] == 'softmax':
        output_activation = tf.nn.softmax
    elif params['output_activation'] == 'relu':
        output_activation = tf.nn.relu
    elif params['output_activation'] == 'tanh':
        output_activation = tf.nn.tanh
    else:
        raise NotImplementedError(
            'Output activation {0} is not valid'.format(
                params['output_activation']))

    hidden_layers = params.get('hidden_layers', [])
    output_dim = params['output_dim']
    dropout = params.get('dropout', None)
    normalizer = params.get('normalizer', None)
    if dp_masks is not None or dropout is None:
        dp_return_masks = None
    else:
        dp_return_masks = []
        distribution = tf.contrib.distributions.Uniform()
    
    dims = hidden_layers + [output_dim]

    next_layer_input = inputs
    with tf.variable_scope(scope, reuse=reuse):
        for i, dim in enumerate(dims):
            if i == len(dims) - 1:
                activation = output_activation
            else:
                activation = hidden_activation
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
            
            if T is None or normalizer != 'batch_norm':   
                next_layer_input = tf.contrib.layers.fully_connected(
                    inputs=next_layer_input,
                    num_outputs=dim,
                    activation_fn=activation,
                    normalizer_fn=normalizer_fn,
                    normalizer_params=normalizer_params,
                    weights_initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
                    biases_initializer=tf.constant_initializer(0., dtype=dtype),
                    weights_regularizer=tf.contrib.layers.l2_regularizer(0.5),
                    trainable=True)
            else:
                fc_out = tf.contrib.layers.fully_connected(
                    inputs=next_layer_input,
                    num_outputs=dim,
                    activation_fn=activation,
                    weights_initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
                    weights_regularizer=tf.contrib.layers.l2_regularizer(0.5),
                    trainable=True)
                fc_out_reshape = tf.reshape(fc_out, (-1, T * fc_out.get_shape()[1].value))
                bn_out = tf.contrib.layers.batch_norm(fc_out_reshape, **normalizer_params)
                next_layer_input = tf.reshape(bn_out, tf.shape(fc_out))

            if dropout is not None:
                assert(type(dropout) is float and 0 < dropout and dropout <= 1.0)
                if dp_masks is not None:
                    next_layer_input = next_layer_input * dp_masks[i]
                else:
                    # Shape is not well defined without reshaping
                    shape = tf.shape(next_layer_input)
                    if num_dp > 1:
                        sample = distribution.sample(shape[0]/num_dp, dim)
                        sample = tf.concat([sample] * num_dp, axis=0)
                    else:
                        sample = distribution.sample(shape)
                    sample = tf.reshape(sample, (-1, dim))
                    mask = tf.cast(sample < dropout, dtype) / dropout
                    next_layer_input = next_layer_input * mask
                    dp_return_masks.append(mask)

        output = next_layer_input  
    
    return output, dp_return_masks    
