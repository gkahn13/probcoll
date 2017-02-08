from general.utility.base_classes import ReadOnlyClass

class iLQRInfo(ReadOnlyClass):
    """ iLQR information container """
    def __init__(self,
                 traj=None, lqr_policy=None,
                 V_approx=None, Q_approx=None,
                 cst_approx=None, dyn_approx=None,
                 cost_func=None, dynamics=None):
        # assert(traj is None or isinstance(traj, Sample))
        # assert(cost_func is None or isinstance(cost_func, Cost))
        # assert(dynamics is None or isinstance(dynamics, Dynamics))
        # assert(V_approx is None or isinstance(V_approx, ValueApprox))
        # assert(cst_approx is None or isinstance(cst_approx, CostApprox))
        # assert(Q_approx is None or isinstance(Q_approx, LocalValueApprox))
        # assert(dyn_approx is None or isinstance(dyn_approx, DynamicsApprox))
        # assert(lqr_policy is None or isinstance(lqr_policy, LinearGaussianPolicy))

        object.__setattr__(self, 'traj', traj)
        object.__setattr__(self, 'V_approx', V_approx)
        object.__setattr__(self, 'Q_approx', Q_approx)
        object.__setattr__(self, 'dynamics', dynamics)
        object.__setattr__(self, 'cost_func', cost_func)
        object.__setattr__(self, 'cst_approx', cst_approx)
        object.__setattr__(self, 'dyn_approx', dyn_approx)
        object.__setattr__(self, 'lqr_policy', lqr_policy)
