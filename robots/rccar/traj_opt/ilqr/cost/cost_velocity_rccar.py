import numpy as np

from rll_quadrotor.planning.ilqr.cost.cost_utils import RAMP_CONSTANT, RAMP_LINEAR, RAMP_QUADRATIC
from rll_quadrotor.planning.ilqr.cost.cost_state import CostState
from rll_quadrotor.planning.ilqr.cost.cost_torque import CostTorque
from rll_quadrotor.planning.ilqr.cost.cost_sum import CostSum
from rll_quadrotor.utility.utils import init_component

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
