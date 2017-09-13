from general.policy.policy_random_planning import PolicyRandomPlanning
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from general.policy.cost.cost_desired import CostDesired
from general.policy.cost.cost_coll import CostColl
from config import params as global_params
import os
try:
    import rospy
    import visualization_msgs.msg as vm
    import geometry_msgs.msg as gm
    import robots.rccar.ros.ros_utils as ros_utils
    use_ros = True
except:
    use_ros = False

class PolicyRandomPlanningBebop2d(PolicyRandomPlanning):
    def __init__(self, probcoll_model, params, dtype=tf.float32):
        PolicyRandomPlanning.__init__(self, probcoll_model, params, dtype=dtype)
        if use_ros and params['visualize']:
            topics = params['topics']
            self.debug_cost_probcoll_pub = ros_utils.Publisher(
                topics['debug_cost_probcoll'],
                vm.MarkerArray,
                queue_size=10)
        self.count = 0
        self._save_dir = os.path.join(global_params['exp_dir'], global_params['exp_name'], 'plots')

    def _setup_action(self):
        with tf.name_scope('random_planner'):
            with self.probcoll_model.graph.as_default():
                control_list = []
                k = self.params['random']['K']
                control_range = self.params['control_range']
                u_distribution = tf.contrib.distributions.Uniform(
                    control_range['lower'],
                    control_range['upper'])
                u_samples = tf.cast(u_distribution.sample(sample_shape=(k, self.probcoll_model.T)), self.dtype)
                if self.params['stop_command']:
                    # stop_command = tf.constant([0., 0., 0.0], dtype='float32', shape=[1, self.probcoll_model.T, 3])
                    temp_command_sequence = [[[0.0, 0.0, 0.0]] * self.probcoll_model.T]
                    command_list = [u_samples,
                                    tf.constant(temp_command_sequence, dtype='float32', shape=[1, self.probcoll_model.T, 3])]
                    for j in xrange(10):
                        temp_command_sequence = [[[0.0, 0.0, 0.0]] * self.probcoll_model.T]
                        for i in xrange(self.probcoll_model.T):
                            # temp_command_sequence[0][i] = self.params['cost']['control_cost']['des']
                            temp_command_sequence[0][i] = [np.random.uniform(low=control_range['lower'][0],
                                                                             high=control_range['upper'][0]),
                                                           np.random.uniform(low=control_range['lower'][1],
                                                                             high=control_range['upper'][1]), 0]
                            command_list.append(tf.constant(temp_command_sequence, dtype='float32', shape=[1, self.probcoll_model.T, 3]))
                    u_samples = tf.concat(command_list, axis=0)
                O_im_input = self.probcoll_model.d_eval['O_im_input']
                O_vec_input = self.probcoll_model.d_eval['O_vec_input']
                stack_u = tf.concat([u_samples] * self.params['num_dp'], axis=0)
                # TODO incorporate std later
                output_pred_mean, output_pred_std, output_mat_mean, output_mat_std = self.probcoll_model.graph_eval_inference(
                    stack_u,
                    O_im_input=O_im_input,
                    O_vec_input=O_vec_input,
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

                control_cost_fn = CostDesired(self.params['cost']['control_cost'])
                # print 'weight for collision: {0}'.format(self.params['cost']['coll_cost']['weight'])
                coll_cost_fn = CostColl(self.params['cost']['coll_cost'])
                control_costs = control_cost_fn.eval(u_samples)
                coll_costs = coll_cost_fn.eval(u_samples, pred_mean, mat_mean, pred_std, mat_std)
                total_cost = control_costs + coll_costs
                index = tf.cast(tf.argmin(total_cost, axis=0), tf.int32)
                action = u_samples[index, 0]
                return action, u_samples, O_im_input, O_vec_input, control_costs, coll_costs

    def visualize(
            self,
            actions_considered,
            action,
            action_noisy,
            coll_costs,
            control_costs):
        # print 'action: {0}'.format(action)
        # print 'action_considered: {0}'.format(actions_considered)
        # print 'coll_costs: {0}'.format(coll_costs)
        # print 'control_costs: {0}'.format(control_costs)
        max_coll_cost = global_params['model']['T'] * self.params['cost']['coll_cost']['weight'][0]
        ns, _ = np.histogram(np.divide(coll_costs, max_coll_cost), bins=20, range=(0, 1))
        self.ns = np.cumsum(np.divide(ns, np.sum(ns), dtype='float32'))
        if self.params['save_fig']:
            if self.params['visualize_probcoll_only']:
                costs = coll_costs
                max_cost = global_params['model']['T'] * self.params['cost']['coll_cost']['weight'][0]
            else:
                costs = coll_costs + control_costs
                max_cost = global_params['model']['T'] * (
                    self.params['cost']['coll_cost']['weight'][0] +
                    self.params['cost']['control_cost']['weight'][0] * (self.params['cost']['control_cost']['des'][0] ** 2))
            temp_indices = np.argsort(costs)
            actions_considered, costs = actions_considered[temp_indices], costs[temp_indices]
            sample_size = self.params['visualize_sample_size']
            sampled_indices = np.linspace(start=0, stop=len(costs) - 1, num=sample_size, dtype='int32')
            actions_considered, costs = actions_considered[sampled_indices], costs[sampled_indices]
            #color_map = cm.ScalarMappable()
            ## standardize
            #colors = color_map.to_rgba(np.divide(costs, max_cost))
            if self.params['visualize_relative_probcoll_cost']:
                colors = cm.inferno(1 - np.divide(costs - np.min(costs), np.max(costs) - np.min(costs)))
            else:
                colors = cm.inferno(1 - np.divide(costs, max_cost))
            # import IPython; IPython.embed()
            plt.figure()
            for i in xrange(len(actions_considered)):
                temp_traj = np.cumsum(actions_considered[i][:, :2], axis=0)
                # if costs[i] < 0.05*max_cost:
                #     plt.plot(temp_traj[:, 1], temp_traj[:, 0], color=[0, 1, 0, 1])
                # else:
                plt.plot(temp_traj[:, 1], temp_traj[:, 0], color=colors[i])
            axis = plt.gca()
            T = global_params['model']['T']
            axis.set_xlim([-self.params['control_range']['upper'][1] * T, self.params['control_range']['upper'][1] * T])
            axis.set_ylim([0, self.params['control_range']['upper'][0] * T])
            if self.params['visualize_relative_probcoll_cost']:
                plt.savefig(self._save_dir + '/' + str(self.count) + '_r.jpg')
            else:
                plt.savefig(self._save_dir + '/' + str(self.count)+'.jpg')
            self.count += 1

