import numpy as np
from cost import Cost
from general.planning.cost.approx import CostApprox
from general.utility.utils import init_component

class CostSum(Cost):
    """
    A wrapper cost function that adds other cost functions
    """
    def __init__(self, no_init=False, **kwargs):
        Cost.__init__(self)
        self._weights = kwargs.pop('weights', [])
        if no_init:
            self._costs = kwargs.pop('costs', [])
        else:
            costs_params = kwargs.pop('costs', [])
            self._costs = [init_component(params) for params in costs_params]
        assert len(self._costs) > 0
        assert isinstance(self._costs[0], Cost)
        assert len(self._costs) == len(self._weights)

    def eval(self, sample, sub_l=None):
        ttl_cst_approx = CostApprox(sample._T, sample._xdim, sample._udim)
        for i in range(len(self._costs)):
            cst_approx = self._costs[i].eval(sample)
            cst_approx *= self._weights[i]
            if isinstance(sub_l, list): sub_l.append(cst_approx.l)
            ttl_cst_approx += cst_approx
        return ttl_cst_approx

    @staticmethod
    def sum_of_costs(weight, *costs):
        costs = [c for c in costs if c is not None]
        assert len(costs) > 0
        if len(costs) == 1:
            return costs[0]
        init_costs = []
        init_weights = []
        for cost in costs:
            assert isinstance(cost, Cost)
            # if isinstance(cost, CostSum):
            #     init_costs.extend(cost._costs)
            # else:
            init_costs.append(cost)
            init_weights.append(1.)
        init_weights = weight * np.array(init_weights)
        return CostSum(no_init=True, weights=init_weights, costs=init_costs)
