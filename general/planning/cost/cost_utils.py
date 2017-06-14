import numpy as np
from general.planning.cost.cost import Cost
from general.state_info.sample import Sample

def finite_differences_cost_test(cost, sample, epsilon=1e-5, threshold=1e-5):
    """
    Check cost function by finite difference.

    :type cost: Cost
    :type sample: Sample
    """
    T = sample._T
    xdim = sample._xdim
    udim = sample._udim
    cst_approx = cost.eval(sample)

    lx_test = np.zeros((T, xdim))
    lu_test = np.zeros((T, udim))
    lxx_test = np.zeros((T, xdim, xdim))
    luu_test = np.zeros((T, udim, udim))
    lux_test = np.zeros((T, udim, xdim))
    for t in range(T):
        x, u = sample.get_X(t=t), sample.get_U(t=t)
        lx_test[t] = finite_differences(
            lambda x: cost.eval_vec(x, u).l, x, (), epsilon
        )
        lu_test[t] = finite_differences(
            lambda u: cost.eval_vec(x, u).l, u, (), epsilon
        )
        lxx_test[t] = finite_differences(
            lambda x: cost.eval_vec(x, u).lx, x, (xdim,), epsilon
        )
        luu_test[t] = finite_differences(
            lambda u: cost.eval_vec(x, u).lu, u, (udim,), epsilon
        )
        lux_test[t] = finite_differences(
            lambda x: cost.eval_vec(x, u).lu, x, (udim,), epsilon
        ).T
    if not np.allclose(cst_approx.lx,  lx_test, atol=threshold):
        print 'lx not close'
    if not np.allclose(cst_approx.lu,  lu_test, atol=threshold):
        print 'lu not close'
    if not np.allclose(cst_approx.lxx, lxx_test, atol=threshold):
        print 'lxx not close'
    if not np.allclose(cst_approx.luu, luu_test, atol=threshold):
        print 'luu not close'
    if not np.allclose(cst_approx.lux, lux_test, atol=threshold):
        print 'lux not close'


def finite_differences(func, inputs, func_output_shape=(), epsilon=1e-5):
    """
    Computes gradients via finite differences.

                   func(x+epsilon)-func(x-epsilon)
    derivative =      ------------------------
                            2*epsilon

    Args:
        func: Function to compute gradient of. Inputs and outputs can be arbitrary dimension.
        inputs (float vector/matrix): Vector value to compute gradient at
        func_output_shape (int tuple, optional): Shape of the output of func. Default is empty-tuple,
            which works for scalar-valued functions.
        epsilon (float, optional): Difference to use for computing gradient.

    Returns:
        Gradient vector of each dimension of func with respect to each dimension of input.
        Will be of shape (inputs_dim X func_output_shape)

    Doctests/Example usages:
    >>> import numpy as np

    #Test vector-shaped gradient
    >>> func = lambda x: x.dot(x)
    >>> g = finite_differences(func, np.array([1.0, 4.0, 9.0]))
    >>> assert np.allclose(g, np.array([2., 8., 18.]))

    #Test matrix-shaped gradient
    >>> func = lambda x: np.sum(x)
    >>> g = finite_differences(func, np.array([[1.0, 2.0], [3.0, 4.0]]))
    >>> assert np.allclose(g, np.array([[ 1.,  1.], [ 1.,  1.]]))

    #Test multi-dim objective function. 2nd derivative of x.dot(x)
    >>> func = lambda x: 2*x
    >>> g = finite_differences(func, np.array([1.0, 2.0]), func_output_shape=(2,))
    >>> assert np.allclose(g, np.array([[ 2.,  0.], [ 0.,  2.]]))
    """
    # assert inputs.ndim == 1 and len(func_output_shape) < 2
    gradient = np.zeros(inputs.shape + func_output_shape)
    for idx, _ in np.ndenumerate(inputs):
        test_input = np.copy(inputs)
        test_input[idx] += epsilon
        obj_d1 = func(test_input)
        obj_d1 = np.squeeze(obj_d1)  # (1,) to float, (1, d) or (d, 1) to (d,)
        # assert obj_d1.shape == func_output_shape
        test_input = np.copy(inputs)
        test_input[idx] -= epsilon
        obj_d2 = func(test_input)
        obj_d2 = np.squeeze(obj_d2)
        # assert obj_d2.shape == func_output_shape
        diff = (obj_d1-obj_d2) / (2*epsilon)
        gradient[idx] += diff
    return gradient


RAMP_CONSTANT = 1
RAMP_LINEAR = 2
RAMP_QUADRATIC = 3
RAMP_FINAL_ONLY = 4
RAMP_QUADRATIC_REV = 5

def get_ramp_multiplier(ramp_option, T, wp_final_multiplier=1.0):
    """
    Returns a time-varying multiplier

    Returns:
        A (T,) float vector containing weights for each timestep
    """
    if not isinstance(ramp_option, int):
        if isinstance(ramp_option, np.ndarray) and ramp_option.size == T:
            wpm = ramp_option
        else:
            raise ValueError('Unknown custom cost ramp!')
    elif ramp_option == RAMP_CONSTANT:
        wpm = np.ones((T,))
    elif ramp_option == RAMP_LINEAR:
        wpm = (np.arange(T, dtype=np.float32)+1) / T
    elif ramp_option == RAMP_QUADRATIC:
        wpm = ((np.arange(T, dtype=np.float32)+1) / T) ** 2
    elif ramp_option == RAMP_FINAL_ONLY:
        wpm = np.zeros((T,))
        wpm[T-1] = 1.0
    elif ramp_option == RAMP_QUADRATIC_REV:
        wpm = 1 - ((np.arange(T, dtype=np.float32)+1) / T) ** 2
    else:
        raise ValueError('Unknown cost ramp requested!')
    wpm[-1] *= wp_final_multiplier
    return wpm


def evall1l2term(wp, d, Jd, Jdd, l1, l2, alpha):
    """
    Evaluate and compute derivatives for combined l1/l2 norm penalty.

    loss = (0.5 * l2 * d^2) + (l1 * sqrt(alpha + d^2))

    Args:
        wp:
            T x D matrix containing weights for each dimension and timestep
        d:
            T x D states to evaluate norm on
        Jd:
            T x D x Dx Jacobian - derivative of d with respect to state
        Jdd:
            T x D x Dx x Dx Jacobian - 2nd derivative of d with respect to state
        l1: l1 loss weight
        l2: l2 loss weight
        alpha:

    Returns:
        l: T, Evaluated loss
        lx: T x Dx First derivative
        lxx: T x Dx x Dx Second derivative
    """
    # Get trajectory length.
    T, _ = d.shape

    # Compute scaled quantities.
    sqrtwp = np.sqrt(wp)
    dsclsq = d * sqrtwp
    dscl = d * wp
    dscls = d * (wp ** 2)

    # Compute total cost.
    l = 0.5 * np.sum(dsclsq ** 2, axis=1) * l2 \
        + np.sqrt(alpha + np.sum(dscl ** 2, axis=1)) * l1

    # First order derivative terms.
    d1 = dscl * l2 + (dscls / np.sqrt(alpha + np.sum(dscl ** 2, axis=1, keepdims=True)) * l1)
    lx = np.sum(Jd * np.expand_dims(d1, axis=2), axis=1)

    # Second order terms.
    psq = np.expand_dims(np.sqrt(alpha + np.sum(dscl ** 2, axis=1, keepdims=True)), axis=1)
    d2 = l1 * ((np.expand_dims(np.eye(wp.shape[1]), axis=0) * (np.expand_dims(wp ** 2, axis=1) / psq)) -
               ((np.expand_dims(dscls, axis=1) * np.expand_dims(dscls, axis=2)) / psq ** 3))
    d2 += l2 * (np.expand_dims(wp, axis=2) * np.tile(np.eye(wp.shape[1]), [T, 1, 1]))

    d1_expand = np.expand_dims(np.expand_dims(d1, axis=-1), axis=-1)
    sec = np.sum(d1_expand * Jdd, axis=1)

    Jd_expand_1 = np.expand_dims(np.expand_dims(Jd, axis=2), axis=4)
    Jd_expand_2 = np.expand_dims(np.expand_dims(Jd, axis=1), axis=3)
    d2_expand = np.expand_dims(np.expand_dims(d2, axis=-1), axis=-1)
    lxx = np.sum(np.sum((Jd_expand_1 * Jd_expand_2) * d2_expand, axis=1), axis=1)

    lxx += 0.5 * sec + 0.5 * np.transpose(sec, [0,2,1])

    return l, lx, lxx


def evallogl2term(wp, d, Jd, Jdd, l1, l2, alpha):
    """
    Evaluate and compute derivatives for combined l1/l2 norm penalty.

    loss = (0.5 * l2 * d^2) + (0.5 * l1 * log(alpha + d^2))

    Args:
        wp:
            T x D matrix containing weights for each dimension and timestep
        d:
            T x D states to evaluate norm on
        Jd:
            T x D x Dx Jacobian - derivative of d with respect to state
        Jdd:
            T x D x Dx x Dx Jacobian - 2nd derivative of d with respect to state
        l1: l1 loss weight
        l2: l2 loss weight
        alpha:

    Returns:
        l: T, Evaluated loss
        lx: T x Dx First derivative
        lxx: T x Dx x Dx Second derivative
    """
    # Get trajectory length.
    T, _ = d.shape

    # Compute scaled quantities.
    sqrtwp = np.sqrt(wp)
    dsclsq = d * sqrtwp
    dscl = d * wp
    dscls = d * (wp ** 2)

    # Compute total cost.
    l = 0.5 * np.sum(dsclsq ** 2, axis=1) * l2 \
        + 0.5*np.log(alpha + np.sum(dscl ** 2, axis=1)) * l1

    # First order derivative terms.
    d1 = dscl * l2 + (dscls / (alpha + np.sum(dscl ** 2, axis=1, keepdims=True)) * l1)
    lx = np.sum(Jd * np.expand_dims(d1, axis=2), axis=1)

    # Second order terms.
    psq = np.expand_dims((alpha + np.sum(dscl ** 2, axis=1, keepdims=True)), axis=1)
    d2 = l1 * ((np.expand_dims(np.eye(wp.shape[1]), axis=0) * (np.expand_dims(wp ** 2, axis=1) / psq)) -
               ((np.expand_dims(dscls, axis=1) * np.expand_dims(dscls, axis=2)) / psq ** 2))
    d2 += l2 * (np.expand_dims(wp, axis=2) * np.tile(np.eye(wp.shape[1]), [T, 1, 1]))

    d1_expand = np.expand_dims(np.expand_dims(d1, axis=-1), axis=-1)
    sec = np.sum(d1_expand * Jdd, axis=1)

    Jd_expand_1 = np.expand_dims(np.expand_dims(Jd, axis=2), axis=4)
    Jd_expand_2 = np.expand_dims(np.expand_dims(Jd, axis=1), axis=3)
    d2_expand = np.expand_dims(np.expand_dims(d2, axis=-1), axis=-1)
    lxx = np.sum(np.sum((Jd_expand_1 * Jd_expand_2) * d2_expand, axis=1), axis=1)

    lxx += 0.5 * sec + 0.5 * np.transpose(sec, [0,2,1])

    return l, lx, lxx
