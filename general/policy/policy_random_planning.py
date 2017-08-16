import tensorflow as tf
from general.policy.policy import Policy
from general.policy.cost.cost_desired import CostDesired
from general.policy.cost.cost_coll import CostColl

class PolicyRandomPlanning(Policy):
    def _setup_action(self):
        with tf.name_scope('random_planner'):
            with self.probcoll_model.graph.as_default():
                control_list = []
                k = self.params['random']['K']
                control_range = self.params['control_range']
                num_dp = self.params['num_dp']
                u_distribution = tf.contrib.distributions.Uniform(
                    control_range['lower'],
                    control_range['upper'])
                u_samples = tf.cast(u_distribution.sample(sample_shape=(k, self.probcoll_model.T)), self.dtype)
                O_im_input = self.probcoll_model.d_eval['O_im_input']
                O_vec_input = self.probcoll_model.d_eval['O_vec_input']
                stack_u = tf.concat([u_samples] * num_dp, axis=0)
                # TODO incorporate std later
                output_pred_mean, output_pred_std, output_mat_mean, output_mat_std = self.probcoll_model.graph_eval_inference(
                    stack_u,
                    O_im_input=O_im_input,
                    O_vec_input=O_vec_input,
                    num_dp=num_dp,
                    reuse=True) 

                pred_mean = tf.reduce_mean(
                    tf.split(output_pred_mean, num_dp, axis=0), axis=0)

                pred_std = tf.reduce_mean(
                    tf.split(output_pred_std, num_dp, axis=0), axis=0)

                mat_mean = tf.reduce_mean(
                    tf.split(output_mat_mean, num_dp, axis=0), axis=0)

                mat_std = tf.reduce_mean(
                    tf.split(output_mat_std, num_dp, axis=0), axis=0)
                
                mat_mean = tf.reduce_mean(
                    tf.split(output_mat_mean, num_dp, axis=0), axis=0)
                
                control_cost_fn = CostDesired(self.params['cost']['control_cost'], self.params['control_range']) 
                coll_cost_fn = CostColl(self.params['cost']['coll_cost'], self.params['control_range'])
                
                control_costs = control_cost_fn.eval(u_samples)
                coll_costs = coll_cost_fn.eval(u_samples, pred_mean, mat_mean, pred_std, mat_std)

                total_cost = control_costs + coll_costs
                avg_cost = tf.reduce_mean(total_cost)
                index = tf.cast(tf.argmin(total_cost, axis=0), tf.int32)
                action = u_samples[index, 0]
                return action, u_samples, O_im_input, O_vec_input, control_costs, coll_costs, avg_cost 
