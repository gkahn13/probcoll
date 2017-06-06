import abc
import tensorflow as tf

from general.tf.planning.noise import ZeroNoise
from general.tf.planning.noise import GaussianNoise
from general.tf.planning.noise import UniformNoise
from general.tf.planning.epsilon_greedy import epsilon_greedy

class Planner(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, probcoll_model, params, dtype=tf.float32):
        self.probcoll_model = probcoll_model
        self.params = params
        self.dtype = self.probcoll_model.dtype
        self._setup()
        self._setup_noise()

    @abc.abstractmethod
    def _setup(self):
        """Needs to define the placeholders self.X_inputs self.O_input and 
        output self.action.
        """
        raise NotImplementedError('Implement in subclass')

    def _setup_noise(self):
        T = self.probcoll_model.T
        noise_type = self.params['control_noise']['type']
        if noise_type == 'Zero':
            self.action_noisy = self.action
        elif noise_type == 'gaussian':
            noise = GaussianNoise(
                self.params['control_noise']['gaussian'],
                dtype=self.dtype) 
            self.action_noisy = self.action + noise
        elif noise_type == 'uniform':
            noise = UniformNoise(
                self.params['control_noise']['uniform'],
                dtype=self.dtype) 
            self.action_noisy = self.action + noise
        else:
            raise NotImplementedError(
                "Noise type {0} is not valid".format(noise_type))
        if self.params['epsilon_greedy']['epsilon'] > 0:
            self.action_noisy = epsilon_greedy(self.action_noisy, self.params['epsilon_greedy'], dtype=self.dtype)

    def plan(self, x, o, t):
        # TODO figure out general way to handle state
        o_input = o[self.probcoll_model.O_idxs()].reshape(1, -1)
        feed_dict = {self.X_inputs: [[[]]*self.probcoll_model.T], self.O_input: o_input}
        action_noisy, action = self.probcoll_model.sess.run(
            [self.action_noisy, self.action],
            feed_dict)
        return action_noisy, action
