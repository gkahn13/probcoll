import abc
import numpy as np

from config import params

class Policy(object):
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
