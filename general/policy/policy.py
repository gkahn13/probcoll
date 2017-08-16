import abc
import tensorflow as tf
import numpy as np

from general.policy.noise import ZeroNoise
from general.policy.noise import GaussianNoise
from general.policy.noise import UniformNoise
from general.policy.epsilon_greedy import epsilon_greedy
from general.utility import schedules

class Policy(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, probcoll_model, params, dtype=tf.float32):
        self.probcoll_model = probcoll_model
        self.params = params
        self.dtype = self.probcoll_model.dtype
        self.reset_ops = []
        with self.probcoll_model.graph.as_default():
            self._t = tf.placeholder(tf.int32, [])
        self.action, self.actions_considered, self.O_im_input, self.O_vec_input,\
                self.control_costs, self.coll_costs, self.avg_cost\
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
            # Noise schedule
            self.noise_schedule = schedules.PiecewiseSchedule(
                endpoints=self.params['control_noise']['endpoints'],
                outside_value=self.params['control_noise']['outside_value'])
            self.noise_multiplier_ph = tf.placeholder(self.dtype, [])
            def get_noisy_action():
                action, _, _, _, _, _, _ = self._setup_action()
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
                return action + (noise * self.noise_multiplier_ph)
            # Epsilon greedy 
            self.eps_schedule = schedules.PiecewiseSchedule(
                endpoints=self.params['epsilon_greedy']['endpoints'],
                outside_value=self.params['epsilon_greedy']['outside_value'])
            self.eps_ph = tf.placeholder(self.dtype, [])
            self.action_noisy = epsilon_greedy(
                get_noisy_action,
                self.params['epsilon_greedy']['control_range']['lower'],
                self.params['epsilon_greedy']['control_range']['upper'],
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

    def get_value(self, obs_frame, t=0):
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
                self._t: t
            }
        avg_cost, = self.probcoll_model.sess.run([self.avg_cost], feed_dict)
        return -1 * avg_cost

    def act(self, obs_frame, t, time_step=0, only_noise=False, only_no_noise=False, visualize=False):
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
                self._t: t
            }
        if not only_no_noise:
            feed_dict[self.eps_ph] = self.eps_schedule.value(time_step)     
            feed_dict[self.noise_multiplier_ph] = self.noise_schedule.value(time_step)
        if visualize:
            action_noisy, action, actions_considered, \
                coll_costs, control_costs  = self.probcoll_model.sess.run(
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
