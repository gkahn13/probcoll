import numpy as np

from general.planning.cost.cost_utils import RAMP_CONSTANT, RAMP_LINEAR, RAMP_QUADRATIC
from general.planning.cost.cost_state import CostState
from general.planning.cost.cost_torque import CostTorque
from general.planning.cost.cost_sum import CostSum
from general.utility.utils import init_component

from config import params

def cost_velocity_pointquad(T, velocity, velocity_weights, weight_scale=1e5):
    costs = []

    cost_velocity = {
        'type': CostTorque,
        'args': {
            'ramp_option': RAMP_CONSTANT,
            'wu': (1e0/T) * np.array(velocity_weights),
            'desired_state': np.array(velocity)
        }
    }
    costs.append(cost_velocity)

    # cost_control = {
    #     'type': CostTorque,
    #     'args': {
    #         'ramp_option': RAMP_CONSTANT,
    #         'wu': (1e0/T) * np.ones(params['U']['dim']),
    #         'desired_state': np.zeros(params['U']['dim'])
    #     }
    # }
    # costs.append(cost_control)

    cost_descr = {
        'type': CostSum,
        'args': {
            'weights': weight_scale * np.ones(len(costs)),
            'costs': costs
        }
    }

    return init_component(cost_descr)
