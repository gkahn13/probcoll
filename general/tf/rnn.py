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
        cell_type = tf.nn.rnn_cell.BasicRNNCell
    elif params["cell_type"] == "mulint_rnn":
        cell_type = rnn_cell.BasicMulintRNNCell
    else:
        raise NotImplementedError(
            "Cell type {0} is not valid".format(params["cell_type"]))

    num_units = params["num_units"]
    num_cells = params["num_cells"]
   
    with tf.variable_scope(scope, reuse=reuse):
        if "cell_args" in params.keys() and  params["cell_args"] is not None:
            cell_args = params["cell_args"]
            cell = cell_type(num_units, **cell_args)
        else:
            cell = cell_type(num_units)
        
        multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_cells)
        outputs, state = tf.nn.dynamic_rnn(
            multi_cell,
            inputs,
            dtype=dtype,
            swap_memory=True,
            time_major=False)

    return outputs, state
