import tensorflow as tf
from general.tf.planning.planner import Planner

import tensorflow as tf
from general.tf.planning.cost.cost_desired import CostDesired
from general.tf.planning.cost.cost_coll import CostColl

class PlannerCem(Planner):
    def _setup(self):
        with tf.name_scope('cem_planner'):
            with self.probcoll_model.graph.as_default():
                control_list = []
                init_m = self.params['cem']['init_M']
                m = self.params['cem']['M']
                k = self.params['cem']['K']
                num_iters = self.params['cem']['num_iters']
                control_range = self.params['control_range']
                d = len(control_range['lower'])
                T = self.probcoll_model.T
                control_cost_fn = CostDesired(self.params['cost']['control_cost']) 
                coll_cost_fn = CostColl(self.params['cost']['coll_cost'])
                self.O_im_input = self.probcoll_model.d_eval['O_im_input']
                self.O_vec_input = self.probcoll_model.d_eval['O_vec_input']

                init_distribution = tf.contrib.distributions.Uniform(
                    control_range['lower'],
                    control_range['upper'])
                init_u_samples = tf.cast(init_distribution.sample(sample_shape=(init_m, T)), self.dtype)
                flat_u_samples = tf.reshape(init_u_samples, (init_m, T * d))
                init_stack_u = tf.concat(0, [init_u_samples]*self.params['num_dp'])
                embeddings = [
                        self.probcoll_model.get_embedding(
                            self.O_im_input,
                            self.O_vec_input,
                            batch_size=1,
                            reuse=True,
                            scope="observation_graph_b{0}".format(b)) for b in xrange(self.probcoll_model.num_bootstrap)
                    ]
                output_pred_mean, output_pred_std, output_mat_mean, output_mat_std = self.probcoll_model.graph_eval_inference(
                    init_stack_u,
                    bootstrap_initial_states=embeddings,
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
                
                init_control_costs = control_cost_fn.eval(init_u_samples)
                init_coll_costs = coll_cost_fn.eval(init_u_samples, pred_mean, mat_mean, pred_std, mat_std)

                total_cost = init_control_costs + init_coll_costs
                
                for _ in xrange(num_iters):
                
                    _, init_top_indices = tf.nn.top_k(-1 * total_cost, k=k)

                    top_controls = tf.gather(flat_u_samples, indices=init_top_indices)

                    flat_top_controls = tf.reshape(top_controls, (k, T * d)) 
                    top_mean = tf.reduce_mean(flat_top_controls, axis=0)
                    top_covar = tf.matmul(tf.transpose(flat_top_controls), flat_top_controls) / k
                    sigma = top_covar + tf.eye(T * d) * self.params['cem']['eps']
                    distribution = tf.contrib.distributions.MultivariateNormalFull(
                        mu=top_mean,
                        sigma=sigma)
                    flat_u_samples_preclip = distribution.sample((m,))
                    flat_u_samples = tf.clip_by_value(
                        flat_u_samples_preclip,
                        control_range['lower'] * T,
                        control_range['upper'] * T)
                    u_samples = tf.reshape(flat_u_samples, (m, T, d))

                    stack_u = tf.concat(0, [u_samples]*self.params['num_dp'])
                    # TODO incorporate std later
                    output_pred_mean, output_pred_std, output_mat_mean, output_mat_std = self.probcoll_model.graph_eval_inference(
                        stack_u,
                        bootstrap_initial_states=embeddings,
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
                    
                    control_costs = control_cost_fn.eval(u_samples)
                    coll_costs = coll_cost_fn.eval(u_samples, pred_mean, mat_mean, pred_std, mat_std)

                    total_cost = control_costs + coll_costs
                    
                index = tf.cast(tf.argmin(total_cost, axis=0), tf.int32)
                self.action = u_samples[index, 0]
