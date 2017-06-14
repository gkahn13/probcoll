import numpy as np
from cost import Cost
from general.planning.cost.approx import CostApprox

class CostTorque(Cost):
    """
    Computes torque penalties
    """

    def __init__(self, **kwargs):
        Cost.__init__(self)
        self._wu = kwargs.pop('wu')
        self._target = np.array(kwargs.pop('desired_state'), dtype=np.float64)

    def eval(self, sample):
        T = sample._T
        udim = sample._udim
        xdim = sample._xdim

        U = sample.get_U()
        cst_approx = CostApprox(T, xdim, udim)
        cst_approx.l = 0.5 * np.sum(self._wu * (U - self._target) ** 2, axis=1)
        cst_approx.lu = self._wu * (U - self._target)
        cst_approx.luu = np.tile(np.diag(self._wu), [T, 1, 1])
        cst_approx.J = np.sum(cst_approx.l)
        return cst_approx
