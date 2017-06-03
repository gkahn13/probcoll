import tensorflow as tf
from general.tf.planning.planner import Planner

import tensorflow as tf
from general.tf.planning.cost.cost_desired import CostDesired
from general.tf.planning.cost.cost_coll import CostColl

class PlannerRandom(Planner):
    def _setup(self):
        # TODO include horizon
        with tf.name_scope('random_planner'):
            control_list = []
            k = self.params['random']['K']
            control_range = self.params['control_range']
            u_distribution = tf.contrib.distributions.Uniform(
                control_range['lower'],
                control_range['upper'])
            u_samples = tf.cast(u_distribution.sample(sample_shape=(k, self.probcoll_model.T)), self.dtype)
            self.X_inputs = self.probcoll_model.d_eval['X_inputs']
            self.O_input = self.probcoll_model.d_eval['O_input']
            stack_u = tf.concat(0, [u_samples]*self.params['num_dp'])
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
            
            control_cost = control_cost_fn.eval(u_samples)
            coll_cost = coll_cost_fn.eval(pred_mean, u_samples)

            total_cost = control_cost + coll_cost
            index = tf.cast(tf.argmin(total_cost, axis=0), tf.int32)
            self.action = u_samples[index, 0]
