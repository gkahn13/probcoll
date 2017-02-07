import numpy as np

from rll_quadrotor.planning.ilqr.cost.cost import Cost
from rll_quadrotor.planning.ilqr.util.approx import CostApprox

class CostDual(Cost):
    """
    l(x_t, u_t) = -u_t^T * lambda_t
    """

    def __init__(self, dual):
        Cost.__init__(self)
        self._dual = dual

    def eval(self, sample):
        T = sample._T
        dX = sample._xdim
        dU = sample._udim
        cst_approx = CostApprox(T, dX, dU)
        for t in range(T):
            cst_approx.l[t] = -self._dual[t].dot(sample.get_U(t=t))
            cst_approx.lu[t] = -self._dual[t]
        cst_approx.J = np.sum(cst_approx.l)
        return cst_approx
