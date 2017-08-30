from general.policy.policy_random_planning import PolicyRandomPlanning
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
from general.policy.cost.cost_desired import CostDesired
from general.policy.cost.cost_coll import CostColl
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
                # stop_command = tf.constant([0., 0., 0.0], dtype='float32', shape=[1, self.probcoll_model.T, 3])
                temp_command_sequence = [[[0.0, 0.0, 0.0]] * self.probcoll_model.T]
                command_list = [u_samples,
                                tf.constant(temp_command_sequence, dtype='float32', shape=[1, self.probcoll_model.T, 3])]
                for i in xrange(self.probcoll_model.T):
                    # temp_command_sequence[0][i] = self.params['cost']['control_cost']['des']
                    temp_command_sequence[0][i] = [np.random.uniform(low=0.6, high=1.0),
                                                   np.random.uniform(low=-0.2, high=0.2), 0]
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
        print 'action_considered: {0}'.format(actions_considered)
        print 'coll_costs: {0}'.format(coll_costs)
        print 'control_costs: {0}'.format(control_costs)
