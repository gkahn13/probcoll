import tensorflow as tf
import abc
import numpy as np
from general.tf.planning.cost.cost_desired import CostDesired
from general.tf.planning.cost.cost_coll import CostColl
from general.tf.planning.planner import Planner


class PlannerPrimitives(Planner):
    __metaclass__ = abc.ABCMeta    

    def __init__(self, probcoll_model, params, dtype=tf.float32):
        self.probcoll_model = probcoll_model
        self.params = params
        self.dtype = self.probcoll_model.dtype
        self._create_primitives()
        self._setup()

    @abc.abstractmethod
    def _create_primitives(self):
        """
        :return tensor of primitive actions
        """
        raise NotImplementedError('Implement in subclass')

    def _setup(self):
        # TODO include horizon
        with tf.name_scope('primitives_planner'):
            self.X_inputs = self.probcoll_model.d_eval['X_inputs']
            self.O_input = self.probcoll_model.d_eval['O_input']
            stack_u = tf.concat(0, [self.primitives]*self.params['num_dp'])
            stack_x = tf.concat(0, [self.X_inputs]*self.params['num_dp'])
            # TODO incorporate std later
            output_pred_mean, _, _, _ = self.probcoll_model.graph_eval_inference(
                stack_x,
                stack_u,
                O_input=self.O_input,
                reuse=True) 

            pred_mean = tf.reduce_mean(
                tf.split(0, self.params['num_dp'], output_pred_mean), axis=0)

            control_cost_fn = CostDesired(self.params['cost']['control_cost']) 
            coll_cost_fn = CostColl(self.params['cost']['coll_cost'])
            
            control_cost = control_cost_fn.eval(self.primitives)
            coll_cost = coll_cost_fn.eval(pred_mean, self.primitives)

            total_cost = control_cost + coll_cost
            index = tf.cast(tf.argmin(total_cost, axis=0), tf.int32)
            self.action = self.primitives[index, 0]

    def plan(self, x, o):
        # TODO figure out general way to handle state
        o_input = o[self.probcoll_model.O_idxs()].reshape(1, -1)
        feed_dict = {self.X_inputs: [[[]]*self.probcoll_model.T], self.O_input: o_input}
        action = self.probcoll_model.sess.run(self.action, feed_dict)
        return action
