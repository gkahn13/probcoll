import tensorflow as tf
from general.policy.cost.cost import Cost

class CostDesired(Cost):
    """Cost for deviating from desired value.
    """

    def __init__(self, params):
        self.des = tf.constant(params['des'])
        self.weight = tf.constant(params['weight'])
        if params['cost'] == "square":
            self.cost = tf.square
        else:
            raise NotImplementedError(
                "CostDesired {0} is not valid".format(cost))

    def eval(self, data):
        diff = data - self.des
        components = self.cost(diff) * self.weight
        weighted_sum = tf.reduce_sum(components, axis=[1, 2])
        
        return weighted_sum
