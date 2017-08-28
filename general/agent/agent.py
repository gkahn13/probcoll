import abc

from general.utility.logger import get_logger
from general.state_info.sample import Sample

from config import params

class Agent(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, dynamics):
        self._dynamics = dynamics
        self._logger = get_logger(self.__class__.__name__, params['world']['logger'])

    def sample_policy(self, x0, policy, T=None, only_noise=False, **policy_args):
        """
        Run the policy and collect the trajectory data

        :param x0: initial state
        :param policy: to execute
        :param policy_args: e.g. ref_traj, noise, etc
        :rtype Sample
        """
        if T is None:
            T = policy._T
        policy_sample = Sample(meta_data=params, T=T)
        policy_sample.set_X(x0, t=0)
        policy_sample_no_noise = Sample(meta_data=params, T=T)
        for t in range(T):
            # get observation and act
            x_t = policy_sample.get_X(t=t)
            o_t = self.get_observation(x_t)
            u_t, u_t_no_noise = policy.act(x_t, o_t, t)
            # record
            policy_sample.set_X(x_t, t=t)
            policy_sample.set_O(o_t, t=t)
            policy_sample.set_U(u_t, t=t)
            if not only_noise:
                policy_sample_no_noise.set_U(u_t_no_noise, t=t)


            # propagate dynamics
            if t < T-1:
                x_tp1 = self._dynamics.evolve(x_t, u_t)
                policy_sample.set_X(x_tp1, t=t+1)

        return policy_sample, policy_sample_no_noise

    @abc.abstractmethod
    def reset(self, x):
        """
        Reset the simulated environment as the specified state.
        Return the actual model state vector.

        :param x: state vector to reset to
        :rtype: np.ndarray
        """
        raise NotImplementedError("Must be implemented in subclass")

    @abc.abstractmethod
    def get_observation(self, x):
        """
        Get observation at state x
        :param x: state vector
        :param noise:
        :return: np.ndarray
        """
        raise NotImplementedError("Must be implemented in subclass")
