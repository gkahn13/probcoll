import tensorflow as tf
from  general.tf import tf_utils
from general.tf import rnn_cell

def rnn(
        inputs,
        initial_state,
        params,
        use_dp_placeholders=False,
        dtype=tf.float32,
        scope="rnn",
        reuse=False):
  
    """
    inputs is shape [batch_size x T x features].
    """
    if params["cell_type"] == "rnn":
        cell_type = rnn_cell.DpRNNCell
        inital_state = (initial_state,)
    elif params["cell_type"] == "mulint_rnn":
        cell_type = rnn_cell.DpMulintRNNCell
        initial_state = (initial_state,)
    else:
        raise NotImplementedError(
            "Cell type {0} is not valid".format(params["cell_type"]))

    num_units = initial_state[0].get_shape()[1].value
    #    num_units = params["num_units"]
    num_cells = params["num_cells"]
    dropout = params.get("dropout", None)
    cell_args = params.get("cell_args", None)
    dropout_placeholders = []
    cells = []

    with tf.variable_scope(scope, reuse=reuse):
        for i in xrange(num_cells):
            if dropout is not None:
                assert(type(dropout) is float and 0 < dropout and dropout < 1.0)
                if use_dp_placeholders:
                    dp = tf.placeholder(dtype, [None, num_units])
                    dropout_placeholders.append(dp)
                else:
                    randoms = tf.random_uniform((num_units,), dtype=dtype)
                    dp = tf.cast(tf.less(randoms, dropout), dtype) / dropout
            else:
                dp = None

            if cell_args is not None:
                cell = cell_type(
                    num_units,
                    dropout_mask=dp,
                    dtype=dtype,
                    num_inputs=inputs.get_shape()[-1],
                    weights_scope="{0}_{1}".format(params["cell_type"], i),
                    **cell_args)
            else:
                cell = cell_type(
                    num_units,
                    dropout_mask=dp,
                    dtype=dtype,
                    num_inputs=inputs.get_shape()[-1],
                    weights_scope="{0}_{1}".format(params["cell_type"], i))
            
            cells.append(cell)
            
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        
        outputs, state = tf.nn.dynamic_rnn(
            multi_cell,
            tf.cast(inputs, dtype),
            initial_state=initial_state,
            dtype=dtype,
            swap_memory=True,
            time_major=False)

    return outputs, dropout_placeholders
