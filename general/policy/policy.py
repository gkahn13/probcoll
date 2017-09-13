import abc
import tensorflow as tf
import numpy as np
import os
from general.policy.noise import ZeroNoise
from general.policy.noise import GaussianNoise
from general.policy.noise import UniformNoise
from general.policy.epsilon_greedy import epsilon_greedy
from general.utility import schedules
from config import params as global_params
class Policy(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, probcoll_model, params, dtype=tf.float32):
        self.probcoll_model = probcoll_model
        self.params = params
        self.dtype = self.probcoll_model.dtype
        self.reset_ops = []
        with self.probcoll_model.graph.as_default():
            self._t = tf.placeholder(tf.int32, [])
        self.action, self.actions_considered, self.O_im_input, \
                self.O_vec_input, self.control_costs, self.coll_costs\
            = self._setup_action()
        self._setup_noise()
        
    @abc.abstractmethod
    def _setup_action(self):
        """Needs to define the placeholders self.O_input and 
        output self.action.
        """
        raise NotImplementedError('Implement in subclass')

    def _setup_noise(self):
        with self.probcoll_model.graph.as_default():
            def get_noisy_action():
                action, _, _, _, _, _ = self._setup_action()
                noise_type = self.params['control_noise']['type']
                if noise_type == 'zero':
                    return action
                elif noise_type == 'gaussian':
                    noise = GaussianNoise(
                        self.params['control_noise']['gaussian'],
                        dtype=self.dtype) 
                elif noise_type == 'uniform':
                    noise = UniformNoise(
                        self.params['control_noise']['uniform'],
                        dtype=self.dtype)
                else:
                    raise NotImplementedError(
                        "Noise type {0} is not valid".format(noise_type))
                return action + noise
                # return noise
            # Epsilon greedy
            self.update_epsilon_greedy_end_points()
            print 'epsilon greedy schedule: {0}'.format(self.params['epsilon_greedy']['endpoints'])
            self.eps_schedule = schedules.PiecewiseSchedule(
                endpoints=self.params['epsilon_greedy']['endpoints'],
                outside_value=self.params['epsilon_greedy']['outside_value'])
            self.eps_ph = tf.placeholder(self.dtype, [])
            self.action_noisy = epsilon_greedy(
                get_noisy_action,
                self.params['control_range']['lower'],
                self.params['control_range']['upper'],
                eps=self.eps_ph,
                dtype=self.dtype)

    def visualize(
            self,
            actions_considered,
            action,
            action_noisy,
            coll_costs,
            control_costs):
        pass

    def update_epsilon_greedy_end_points(self):
        temp_dir = global_params['exp_dir'] + '/' + global_params['exp_name']
        num_ite_completed = len(filter(lambda x: x.startswith('itr'), os.listdir(temp_dir)))
        rollout_per_itr = global_params['probcoll']['num_rollouts']
        num_rollouts_completed = num_ite_completed* rollout_per_itr
        end_points = self.params['epsilon_greedy']['endpoints']
        points_passed = 0
        if num_rollouts_completed > 0:
            if len(end_points) > 1:
                for point in end_points:
                    point[0] -= num_rollouts_completed
                    if point[0] <= 0:
                        points_passed += 1
                gradients = []
                for i in xrange(len(end_points)-1):
                    gradients.append((end_points[i+1][1] - end_points[i][1])/
                                     float(end_points[i+1][0]-end_points[i][0]))
                if points_passed == len(end_points):
                    self.params['epsilon_greedy']['endpoints'] = [[0, 0.0]]
                else:
                    for i in xrange(len(end_points)-1):
                        if end_points[i][0] <= 0 and end_points[i+1][0] > 0:
                            end_points[i] = [0, end_points[i][1] + gradients[i]*(0-end_points[i][0])]
                    new_end_points = []
                    for point in end_points:
                        if point[0] >= 0:
                            new_end_points.append(point)
    def act(self, obs_frame, t, rollout_num, only_noise=False, only_no_noise=False, visualize=False):
        assert(not only_noise or not only_no_noise)
        if t == 0:
            self.probcoll_model.sess.run(self.reset_ops)
        o_im_input = []
        o_vec_input = []
        for o in obs_frame:
            o_im_input.append(o[self.probcoll_model.O_im_idxs()])
            o_vec_input.append(o[self.probcoll_model.O_vec_idxs()])
        o_im_input = np.concatenate(o_im_input).reshape(1, -1)
        o_vec_input = np.concatenate(o_vec_input).reshape(1, -1)
        feed_dict = {
                self.O_im_input: o_im_input.astype(np.uint8),
                self.O_vec_input: o_vec_input,
                self.eps_ph: self.eps_schedule.value(rollout_num),
                self._t: t
            }
        if visualize:
            action_noisy, action, actions_considered, \
                coll_costs, control_costs = self.probcoll_model.sess.run(
                    [
                        self.action_noisy,
                        self.action,
                        self.actions_considered,
                        self.coll_costs,
                        self.control_costs
                    ],
                    feed_dict)
            self.visualize(
                actions_considered,
                action,
                action_noisy,
                coll_costs,
                control_costs)
        else:
            if only_noise:
                action_noisy = self.probcoll_model.sess.run(
                    self.action_noisy,
                    feed_dict)
                action=None
            elif only_no_noise:
                action = self.probcoll_model.sess.run(
                    self.action,
                    feed_dict)
                action_noisy = None
            else:
                action_noisy, action = self.probcoll_model.sess.run(
                    [self.action_noisy, self.action],
                    feed_dict)
        return action_noisy, action
