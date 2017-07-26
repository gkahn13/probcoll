import tensorflow as tf
import numpy as np
from general.policy.policy import Policy
from general.policy.cost.cost_desired import CostDesired
from general.policy.cost.cost_coll import CostColl

class PolicyCem(Policy):
    def _dist_eval(self, distribution, embeddings, control_cost_fn, coll_cost_fn, m, T, dU):
        flat_u_samples_preclip = distribution.sample((m,))
        flat_u_samples = tf.clip_by_value(
            flat_u_samples_preclip,
            np.array(self.params['control_range']['lower'] * T, dtype=np.float32),
            np.array(self.params['control_range']['upper'] * T, dtype=np.float32))
        u_samples = tf.cast(tf.reshape(flat_u_samples, (m, T, dU)), dtype=self.dtype)
        num_dp = self.params['num_dp']
        stack_u = tf.concat([u_samples] * num_dp, axis=0)
        # TODO incorporate std later
        output_pred_mean, output_pred_std, output_mat_mean, output_mat_std = self.probcoll_model.graph_eval_inference(
            stack_u,
            bootstrap_initial_states=embeddings,
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
        
        control_costs = control_cost_fn.eval(u_samples)
        coll_costs = coll_cost_fn.eval(u_samples, pred_mean, mat_mean, pred_std, mat_std)

        total_cost = control_costs + coll_costs
        return u_samples, total_cost, control_costs, coll_costs 

    def _cem_fn(self, params, init_distribution, embeddings, control_cost_fn, coll_cost_fn, control_range, eps):
        def _cem():
            init_m = params['init_M']
            m = params['M']
            k = params['K']
            num_iters = params['num_additional_iters']
            dU = len(control_range['lower'])
            T = self.probcoll_model.T
            u_samples, total_cost, control_costs, coll_costs = self._dist_eval(
                init_distribution,
                embeddings,
                control_cost_fn,
                coll_cost_fn,
                init_m,
                T,
                dU)

            flat_u_samples = tf.reshape(u_samples, (init_m, -1))
            for _ in xrange(num_iters):
            
                _, top_indices = tf.nn.top_k(-1 * total_cost, k=k)

                top_controls = tf.gather(flat_u_samples, indices=top_indices)

                flat_top_controls = tf.reshape(top_controls, (k, T * dU)) 
                top_mean = tf.reduce_mean(flat_top_controls, axis=0)
                top_covar = tf.matmul(tf.transpose(flat_top_controls), flat_top_controls) / k
                sigma = top_covar + tf.eye(T * dU) * eps
                distribution = tf.contrib.distributions.MultivariateNormalFullCovariance(
                    loc=top_mean,
                    covariance_matrix=sigma)
                u_samples, total_cost, control_costs, coll_costs = self._dist_eval(
                    distribution,
                    embeddings,
                    control_cost_fn,
                    coll_cost_fn,
                    m,
                    T,
                    dU) 

            index = tf.cast(tf.argmin(total_cost, axis=0), tf.int32)
            action_seq = u_samples[index]
            return action_seq, u_samples, control_costs, coll_costs
        return _cem

    def _setup_action(self):
        with tf.name_scope('cem_planner'):
            with self.probcoll_model.graph.as_default():
                O_im_input = self.probcoll_model.d_eval['O_im_input']
                O_vec_input = self.probcoll_model.d_eval['O_vec_input']
                embeddings = [
                        self.probcoll_model.get_embedding(
                            O_im_input,
                            O_vec_input,
                            batch_size=1,
                            reuse=True,
                            scope="observation_graph_b{0}".format(b)) for b in xrange(self.probcoll_model.num_bootstrap)
                    ]
                control_cost_fn = CostDesired(self.params['cost']['control_cost']) 
                coll_cost_fn = CostColl(self.params['cost']['coll_cost'])

                T = self.probcoll_model.T
                eps = self.params['cem']['eps']
                control_range = self.params['control_range']
                dU = len(control_range['lower'])
                control_lower = np.array(control_range['lower'] * T, dtype=np.float32)
                control_upper = np.array(control_range['upper'] * T, dtype=np.float32)
                control_std = np.square(control_upper - control_lower) / 12.0

                reuse = hasattr(self, 'action')
                with tf.variable_scope('cem_warm_start', reuse=reuse):
                    mu = tf.get_variable('mu', [dU * T], trainable=False)
                self.reset_ops.append(mu.initializer)

                init_distribution = tf.contrib.distributions.Uniform(
		    control_lower,
                    control_upper)
                ws_distribution = tf.contrib.distributions.MultivariateNormalDiag(
                    loc=mu,
                    scale_diag=control_std)

                action_seq, u_samples, control_costs, coll_costs = tf.cond(
                    tf.greater(self._t, 0),
                    self._cem_fn(
                        self.params['cem']['warm_start'],
                        ws_distribution,
                        embeddings,
                        control_cost_fn,
                        coll_cost_fn,
                        control_range,
                        eps),
                    self._cem_fn(
                        self.params['cem'],
                        init_distribution,
                        embeddings,
                        control_cost_fn,
                        coll_cost_fn,
                        control_range,
                        eps))

                flat_end_action_seq = tf.reshape(action_seq[1:], (dU * (T - 1),))
                next_mean = tf.concat([flat_end_action_seq, flat_end_action_seq[-dU:]], axis=0)
                update_mean = tf.assign(mu, next_mean)
                with tf.control_dependencies([update_mean]):
                    action = action_seq[0]
                return action, u_samples, O_im_input, O_vec_input, control_costs, coll_costs 
