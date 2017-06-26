import tensorflow as tf
from  general.tf import tf_utils
from general.tf.nn import rnn_cell

def rnn(
        inputs,
        params,
        initial_state=None,
        dp_masks=None,
        dtype=tf.float32,
        scope="rnn",
        reuse=False):
  
    """
    inputs is shape [batch_size x T x features].
    """
    # TODO adjust state for Mulicell
    if params["cell_type"] == "rnn":
        cell_type = rnn_cell.DpRNNCell
        if initial_state is not None:
            num_units = initial_state.get_shape()[1].value
            inital_state = (initial_state,)
    elif params["cell_type"] == "mulint_rnn":
        cell_type = rnn_cell.DpMulintRNNCell
        if initial_state is not None:
            num_units = initial_state.get_shape()[1].value
            initial_state = (initial_state,)
    elif params['cell_type'] == 'lstm':
        cell_type = rnn_cell.DpLSTMCell
        # TODO maybe change
        if initial_state is not None:
            c, h = tf.split(1, 2, initial_state)
            num_units = c.get_shape()[1].value
            initial_state = (tf.nn.rnn_cell.LSTMStateTuple(c, h),)
    elif params['cell_type'] == 'mulint_lstm':
        cell_type = rnn_cell.DpMulintLSTMCell
        if initial_state is not None:
            c, h = tf.split(1, 2, initial_state)
            num_units = c.get_shape()[1].value
            initial_state = (tf.nn.rnn_cell.LSTMStateTuple(c, h),)
    else:
        raise NotImplementedError(
            "Cell type {0} is not valid".format(params["cell_type"]))

    if initial_state is None:
        num_units = params["num_units"]
    num_cells = params["num_cells"]
    dropout = params.get("dropout", None)
    cell_args = params.get("cell_args", {})
    if dp_masks is not None or dropout is None:
        dp_return_masks = None
    else:
        dp_return_masks = []
        distribution = tf.contrib.distributions.Uniform()
    cells = []

    with tf.variable_scope(scope, reuse=reuse):
        for i in xrange(num_cells):
            if dropout is not None:
                assert(type(dropout) is float and 0 < dropout and dropout <= 1.0)
                if dp_masks is not None:
                    dp = dp_masks[i]
                else:
                    # Shape is not well defined without reshaping
                    sample = tf.reshape(distribution.sample((tf.shape(inputs)[0], num_units)), (-1, num_units))
                    mask = tf.cast(sample < dropout, dtype) / dropout
                    dp = mask
                    dp_return_masks.append(mask)
            else:
                dp = None

            cell = cell_type(
                num_units,
                dropout_mask=dp,
                dtype=dtype,
                num_inputs=inputs.get_shape()[-1],
                weights_scope="{0}_{1}".format(params["cell_type"], i),
                **cell_args)
            
            cells.append(cell)
            
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        outputs, state = tf.nn.dynamic_rnn(
            multi_cell,
            tf.cast(inputs, dtype),
            initial_state=initial_state,
            dtype=dtype,
            time_major=False)
    
    return outputs, dp_return_masks
