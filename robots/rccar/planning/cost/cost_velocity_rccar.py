import numpy as np

from general.planning.cost.cost_utils import RAMP_CONSTANT, RAMP_LINEAR, RAMP_QUADRATIC
from general.planning.cost.cost_state import CostState
from general.planning.cost.cost_torque import CostTorque
from general.planning.cost.cost_sum import CostSum
from general.utility.utils import init_component

def cost_velocity_rccar(T, u_des, u_weights, weight_scale=1e0):
    costs = []

    cost_control = {
        'type': CostTorque,
        'args': {
            'ramp_option': RAMP_CONSTANT,
            'wu': (1e0/T) * np.array(u_weights),
            'desired_state': np.array(u_des)
        }
    }
    costs.append(cost_control)

    cost_descr = {
        'type': CostSum,
        'args': {
            'weights': weight_scale * np.ones(len(costs)),
            'costs': costs
        }
    }

    return init_component(cost_descr)
