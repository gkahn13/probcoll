import tensorflow as tf
import numpy as np
from general.tf.planning.planner import Planner
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
                dU = len(control_range['lower'])
                T = self.probcoll_model.T
                control_cost_fn = CostDesired(self.params['cost']['control_cost']) 
                coll_cost_fn = CostColl(self.params['cost']['coll_cost'])
                self.O_im_input = self.probcoll_model.d_eval['O_im_input']
                self.O_vec_input = self.probcoll_model.d_eval['O_vec_input']
                control_lower = np.array(control_range['lower'] * T, dtype=np.float32)
                control_upper = np.array(control_range['upper'] * T, dtype=np.float32)
                control_mean = (control_upper + control_lower) / 2.0
                # TODO figure out what to set std
                control_std = np.square(control_upper - control_lower) / 12.0
                self.mu = tf.get_variable('mu', [dU * T,], initializer=tf.constant_initializer(control_mean, dtype=self.dtype), trainable=False)  
                self.reset_ops.append(self.mu.initializer)
                self.diag_std = tf.get_variable('diag_std', [dU * T,], initializer=tf.constant_initializer(control_std, dtype=self.dtype), trainable=False)
                self.reset_ops.append(self.diag_std.initializer)
                init_distribution = tf.contrib.distributions.MultivariateNormalDiag(
                    loc=self.mu,
                    scale_diag=self.diag_std)
                flat_u_samples = tf.cast(init_distribution.sample(sample_shape=(init_m,)), self.dtype)
                u_samples = tf.reshape(flat_u_samples, (init_m, T, dU))
                init_stack_u = tf.concat([u_samples]*self.params['num_dp'], axis=0)
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
                    tf.split(output_pred_mean, self.params['num_dp'], axis=0), axis=0)

                pred_std = tf.reduce_mean(
                    tf.split(output_pred_std, self.params['num_dp'], axis=0), axis=0)

                mat_mean = tf.reduce_mean(
                    tf.split(output_mat_mean, self.params['num_dp'], axis=0), axis=0)

                mat_std = tf.reduce_mean(
                    tf.split(output_mat_std, self.params['num_dp'], axis=0), axis=0)
                
                mat_mean = tf.reduce_mean(
                    tf.split(output_mat_mean, self.params['num_dp'], axis=0), axis=0)
                
                init_control_costs = control_cost_fn.eval(u_samples)
                init_coll_costs = coll_cost_fn.eval(u_samples, pred_mean, mat_mean, pred_std, mat_std)

                total_cost = init_control_costs + init_coll_costs
                
                for _ in xrange(num_iters):
                
                    _, init_top_indices = tf.nn.top_k(-1 * total_cost, k=k)

                    top_controls = tf.gather(flat_u_samples, indices=init_top_indices)

                    flat_top_controls = tf.reshape(top_controls, (k, T * dU)) 
                    top_mean = tf.reduce_mean(flat_top_controls, axis=0)
                    top_covar = tf.matmul(tf.transpose(flat_top_controls), flat_top_controls) / k
                    sigma = top_covar + tf.eye(T * dU) * self.params['cem']['eps']
                    distribution = tf.contrib.distributions.MultivariateNormalFullCovariance(
                        loc=top_mean,
                        covariance_matrix=sigma)
                    flat_u_samples_preclip = distribution.sample((m,))
                    flat_u_samples = tf.clip_by_value(
                        flat_u_samples_preclip,
                        control_lower,
                        control_upper)
                    u_samples = tf.reshape(flat_u_samples, (m, T, dU))

                    stack_u = tf.concat([u_samples]*self.params['num_dp'], axis=0)
                    # TODO incorporate std later
                    output_pred_mean, output_pred_std, output_mat_mean, output_mat_std = self.probcoll_model.graph_eval_inference(
                        stack_u,
                        bootstrap_initial_states=embeddings,
                        reuse=True) 

                    pred_mean = tf.reduce_mean(
                        tf.split(output_pred_mean, self.params['num_dp'], axis=0), axis=0)

                    pred_std = tf.reduce_mean(
                        tf.split(output_pred_std, self.params['num_dp'], axis=0), axis=0)

                    mat_mean = tf.reduce_mean(
                        tf.split(output_mat_mean, self.params['num_dp'], axis=0), axis=0)

                    mat_std = tf.reduce_mean(
                        tf.split(output_mat_std, self.params['num_dp'], axis=0), axis=0)
                    
                    mat_mean = tf.reduce_mean(
                        tf.split(output_mat_mean, self.params['num_dp'], axis=0), axis=0)
                    
                    control_costs = control_cost_fn.eval(u_samples)
                    coll_costs = coll_cost_fn.eval(u_samples, pred_mean, mat_mean, pred_std, mat_std)

                    total_cost = control_costs + coll_costs
                    
                index = tf.cast(tf.argmin(total_cost, axis=0), tf.int32)
                # For warm start
                action_seq = u_samples[index]
                flat_end_action_seq = tf.reshape(action_seq[1:], (dU * (T - 1),))
                next_mean = tf.concat([flat_end_action_seq, flat_end_action_seq[-dU:]], axis=0)
                next_std = tf.constant(control_std / 4.)
                update_mean = tf.assign(self.mu, next_mean)
                update_std = tf.assign(self.diag_std, next_std)
                with tf.control_dependencies([update_mean, update_std]):
                    self.action = action_seq[0]
