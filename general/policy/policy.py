__all__ = ['Policy', 'GaussianPolicy', 'LinearPolicy']

import abc
import numpy as np
from general.utility.base_classes import UncopyableClass

from config import params

class Policy(UncopyableClass):
    """
    Computes actions from states/observations
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def act(self, x, obs, t):
        """
        Return the action under this policy given current state or observation
        """
        raise NotImplementedError()

    def get_info(self):
        """
        Keeps track of relevant info for saving
        """
        return dict()


class GaussianPolicy(object):
    def get_noise(self, noise):
        if noise is True:
            noise = np.random.normal(0., 1., params['U']['dim'])
        if noise is None or noise is False:  # None, False, 0
            noise = np.zeros((params['U']['dim'],))
        return noise


class LinearPolicy(object):
    """
    Interface for linear policy or policy that can be linearized
    """
    def linearize(self, x=None, u=None, t=None):
        """
        Linearize this policy with given reference point
        i.e. act(x, u) = u_ref + k + K * (x - x_ref)


        :return: k of size (dU,), K of size (dU, dX)
        :rtype: (np.ndarray, np.ndarray)
        """
        raise NotImplementedError('Interface method')

    def linearize_all(self, traj):
        """
        Linearize this policy with given trajectory

        :type traj: Sample
        :return: k of size (H, dU), K of size (H, dU, dX)
        :rtype: (np.ndarray, np.ndarray)
        """
        raise NotImplementedError('Interface method')
