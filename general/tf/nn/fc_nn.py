from general.tf import tf_utils
import tensorflow as tf

#TODO batch norm
def fcnn(
        inputs,
        params,
        dp_masks=None,
        dtype=tf.float32,
        scope="fcnn",
        reuse=False,
        is_training=True):
    
    if "hidden_activation" not in params:
        hidden_activation = None
    elif params["hidden_activation"] == "relu":
        hidden_activation = tf.nn.relu
    elif params["hidden_activation"] == "tanh":
        hidden_activation = tf.nn.tanh
    else:
        raise NotImplementedError(
            "Hidden activation {0} is not valid".format(
                params["hidden_activation"]))

    if "output_activation" not in params:
        output_activation = None
    elif params["output_activation"] == "sigmoid":
        output_activation = tf.nn.sigmoid
    elif params["output_activation"] == "softmax":
        output_activation = tf.nn.softmax
    elif params["output_activation"] == "relu":
        output_activation = tf.nn.relu
    elif params["output_activation"] == "tanh":
        output_activation = tf.nn.tanh
    else:
        raise NotImplementedError(
            "Output activation {0} is not valid".format(
                params["output_activation"]))

    hidden_layers = params.get("hidden_layers", [])
    output_dim = params["output_dim"]
    dropout = params.get("dropout", None)
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
            next_layer_input = tf.contrib.layers.fully_connected(
                inputs=next_layer_input,
                num_outputs=dim,
                activation_fn=activation,
                weights_initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
                biases_initializer=tf.constant_initializer(0., dtype=dtype),
                weights_regularizer=tf.contrib.layers.l2_regularizer(0.5),
                trainable=True)

            if dropout is not None:
                assert(type(dropout) is float and 0 < dropout and dropout <= 1.0)
                if dp_masks is not None:
                    next_layer_input = next_layer_input * dp_masks[i]
                else:
                    # Shape is not well defined without reshaping
                    sample = tf.reshape(distribution.sample(tf.shape(next_layer_input)), (-1, dim))
                    mask = tf.cast(sample < dropout, dtype) / dropout
                    next_layer_input = next_layer_input * mask
                    dp_return_masks.append(mask)

        output = next_layer_input  
    
    return output, dp_return_masks    
