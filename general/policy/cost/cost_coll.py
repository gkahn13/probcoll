import tensorflow as tf
import numpy as np
from general.policy.cost.cost import Cost

class CostColl(Cost):
    """Cost for collision.
    """

    def __init__(self, params, control_range):
        self.weight = tf.constant(params['weight'])
        self.std_weight = params.get('std_weight', 0.)
        self.range = np.array(control_range['upper'])
        self.pre_activation = params.get('pre_activation', False)
        if params['cost'] == "square":
            self.cost = tf.square
        else:
            raise NotImplementedError(
                "CostColl {0} is not valid".format(cost))

    def eval(self, data, pred_mean=None, mat_mean=None, pred_std=None, mat_std=None):
        if self.std_weight > 0:
            if self.pre_activation:
                probcoll = tf.nn.sigmoid(mat_mean + std_weight * mat_std) 
            else:
                probcoll = pred_mean + std_weight * pred_std
        else:
            probcoll = pred_mean
        components = self.cost(data / self.range) * probcoll
        weighted_sum = tf.reduce_sum(components * self.weight, axis=(1, 2))
        return weighted_sum
