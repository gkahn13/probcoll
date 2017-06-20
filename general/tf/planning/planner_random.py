import tensorflow as tf
from general.tf.planning.planner import Planner
from general.tf.planning.cost.cost_desired import CostDesired
from general.tf.planning.cost.cost_coll import CostColl

class PlannerRandom(Planner):
    def _setup(self):
        with tf.name_scope('random_planner'):
            control_list = []
            k = self.params['random']['K']
            control_range = self.params['control_range']
            u_distribution = tf.contrib.distributions.Uniform(
                control_range['lower'],
                control_range['upper'])
            u_samples = tf.cast(u_distribution.sample(sample_shape=(k, self.probcoll_model.T)), self.dtype)
            self.actions_considered = u_samples
            self.O_im_input = self.probcoll_model.d_eval['O_im_input']
            self.O_vec_input = self.probcoll_model.d_eval['O_vec_input']
            stack_u = tf.concat(0, [u_samples]*self.params['num_dp'])
            # TODO incorporate std later
            output_pred_mean, output_pred_std, output_mat_mean, output_mat_std = self.probcoll_model.graph_eval_inference(
                stack_u,
                O_im_input=self.O_im_input,
                O_vec_input=self.O_vec_input,
                reuse=True) 

            pred_mean = tf.reduce_mean(
                tf.split(0, self.params['num_dp'], output_pred_mean), axis=0)

            pred_std = tf.reduce_mean(
                tf.split(0, self.params['num_dp'], output_pred_std), axis=0)

            mat_mean = tf.reduce_mean(
                tf.split(0, self.params['num_dp'], output_mat_mean), axis=0)

            mat_std = tf.reduce_mean(
                tf.split(0, self.params['num_dp'], output_mat_std), axis=0)
            
            mat_mean = tf.reduce_mean(
                tf.split(0, self.params['num_dp'], output_mat_mean), axis=0)
            
            control_cost_fn = CostDesired(self.params['cost']['control_cost']) 
            coll_cost_fn = CostColl(self.params['cost']['coll_cost'])
            
            self.control_costs = control_cost_fn.eval(u_samples)
            self.coll_costs = coll_cost_fn.eval(u_samples, pred_mean, mat_mean, pred_std, mat_std)

            total_cost = self.control_costs + self.coll_costs
            index = tf.cast(tf.argmin(total_cost, axis=0), tf.int32)
            self.action = u_samples[index, 0]
