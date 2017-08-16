import tensorflow as tf
import numpy as np
from general.policy.cost.cost import Cost

class CostDesired(Cost):
    """Cost for deviating from desired value.
    """

    def __init__(self, params, control_range):
        self.des = tf.constant(params['des'])
        self.weight = tf.constant(params['weight'])
        ran = np.array(control_range['upper']) - np.array(control_range['lower'])
        self.range = np.clip(ran, a_min=1.e-3, a_max=np.inf)
        if params['cost'] == "square":
            self.cost = tf.square
        else:
            raise NotImplementedError(
                "CostDesired {0} is not valid".format(cost))

    def eval(self, data):
        diff = (data - self.des) / self.range
        components = self.cost(diff) * self.weight
        weighted_sum = tf.reduce_sum(components, axis=[1, 2])
        
        return weighted_sum
