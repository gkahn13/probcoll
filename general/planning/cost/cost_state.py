import numpy as np
from cost import *
from cost_utils import evall1l2term, get_ramp_multiplier, RAMP_CONSTANT

class CostState(Cost):
    """
    Computes l1/l2 distance to a fixed target state
    """

    def __init__(self, **kwargs):
        Cost.__init__(self)
        self._l1 = kwargs.pop('l1', 0.0)
        self._l2 = kwargs.pop('l2', 1.0)
        self._alpha = kwargs.pop('alpha', 1e-2)
        self._wp_final_mult = kwargs.pop('wp_final_mult', 1.0)
        self._ramp_option = kwargs.pop('ramp_option', RAMP_CONSTANT)
        self._types_info = kwargs.pop('data_types', {})

    def eval(self, sample):
        T = sample._T
        udim = sample._udim
        xdim = sample._xdim

        final_l = np.zeros(T)
        final_lu = np.zeros((T, udim))
        final_lx = np.zeros((T, xdim))
        final_luu = np.zeros((T, udim, udim))
        final_lxx = np.zeros((T, xdim, xdim))
        final_lux = np.zeros((T, udim, xdim))

        for data_type, config in self._types_info.items():
            dS = sample.get_X_dim(sub_state=data_type)
            dSidxs = sample.get_X_idxs(sub_state=data_type)

            # compute wpm according to T to slice T of interest
            wp = np.array(config['wp'], dtype=np.float64)
            wpm = get_ramp_multiplier(
                self._ramp_option, T,
                wp_final_multiplier=self._wp_final_mult
            )
            wp = wp*np.expand_dims(wpm, axis=-1)

            # Compute state penalty
            tgt = np.array(config['desired_state'], dtype=np.float64)
            X = sample.get_X(sub_state=data_type)
            dist = X - tgt

            if 'inv_sqrt_sigmas' in config.keys():
                inv_sqrt_sigmas = config['inv_sqrt_sigmas']
            else:
                inv_sqrt_sigmas = np.tile(np.eye(dS), [T, 1, 1])

            # Evaluate penalty term.
            l, ls, lss = evall1l2term(
                wp,
                dist,
                inv_sqrt_sigmas,
                np.zeros((T, dS, dS, dS)),
                self._l1, self._l2, self._alpha
            )

            final_l += l
            final_lx[:,dSidxs] = ls
            final_lxx[:,dSidxs,dSidxs] = lss

        J = np.sum(final_l)

        cst_approx = CostApprox(T, xdim, udim)
        cst_approx.J = J
        cst_approx.l = final_l
        cst_approx.lx = final_lx
        cst_approx.lxx = final_lxx
        cst_approx.lu = final_lu
        cst_approx.luu = final_luu
        cst_approx.lux = final_lux
        return cst_approx
