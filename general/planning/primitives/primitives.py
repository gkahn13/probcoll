import abc

import numpy as np

from general.planning.planner import Planner

class Primitives(Planner):
    __metaclass__ = abc.ABCMeta

    def __init__(self, H, dynamics, cost_funcs, use_mpc):
        Planner.__init__(self, H, dynamics, cost_funcs, use_mpc)
        self._primitives = self._create_primitives()

    @abc.abstractmethod
    def _create_primitives(self):
        """
        :return: list of Sample in which only the U are filled in
        """
        raise NotImplementedError('Implement in subclass')

    def plan(self, x, o):
        assert(np.isfinite(x).all())
        assert(np.isfinite(o).all())

        ### rollout each primitive
        for i, primitive in enumerate(self._primitives):
            if i == 0:
                primitive.set_O(o, t=0)
            primitive.set_X(x, t=0)
            primitive.rollout(self._dynamics)

        costs = np.zeros(len(self._primitives), dtype=float)
        ### evaluate cost of each primitive
        for cost_func in self._cost_funcs:
            costs += [cst_approx.J for cst_approx in cost_func.eval_batch(self._primitives)]

        return self._primitives[np.argmin(costs)].copy()

    def _mpc_update(self):
        pass
