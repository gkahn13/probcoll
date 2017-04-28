import tensorflow as tf
from general.tf import rnn_cell

def rnn(
        inputs,
        params,
        use_dp_placeholders=False, #TODO
        scope="rnn",
        reuse=False):
  
    """
    inputs is shape [batch_size x T x features].
    """
    #TODO
    dtype = tf.float32
    if params["cell_type"] == "rnn":
        cell_type = rnn_cell.DpRNNCell
    elif params["cell_type"] == "mulint_rnn":
        cell_type = rnn_cell.DpMulintRNNCell
    else:
        raise NotImplementedError(
            "Cell type {0} is not valid".format(params["cell_type"]))

    num_units = params["num_units"]
    num_cells = params["num_cells"]
    dropout = params["dropout"]
    dropout_placeholders = []
    cells = []

    with tf.variable_scope(scope, reuse=reuse):
        for i in xrange(num_cells):
            if dropout is not None and dropout < 1.0:
                assert(type(dropout) is float and 0 <= dropout and dropout <= 1.0)
                if use_dp_placeholders:
                    dp = tf.placeholder(tf.float32, [None, num_units])
                    dropout_placeholders.append(dp)
                else:
                    randoms = tf.random_uniform((num_units,), dtype=dtype)
                    dp = tf.cast(tf.less(randoms, dropout), dtype) / dropout
            else:
                dp = None

            if "cell_args" in params.keys() and  params["cell_args"] is not None:
                cell_args = params["cell_args"]
                cell = cell_type(num_units, dropout_mask=dp, **cell_args)
            else:
                cell = cell_type(num_units, dropout_mask=dp)
            
            cells.append(cell)
            
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        outputs, state = tf.nn.dynamic_rnn(
            multi_cell,
            inputs,
            dtype=dtype,
            swap_memory=True,
            time_major=False)

    return outputs, dropout_placeholders
