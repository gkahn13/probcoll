import abc

from rll_quadrotor.utility.logger import get_logger

from config import params

class World(object):

    def __init__(self, wp=None):
        """
        :param wp: world params
        """
        self._logger = get_logger(self.__class__.__name__, 'info')

        self.wp = wp if wp is not None else params['world']

    @abc.abstractmethod
    def reset(self, cond=None, itr=None):
        raise NotImplementedError('Implement in subclass')

    @abc.abstractmethod
    def is_collision(self, sample):
        raise NotImplementedError('Implement in subclass')

    @abc.abstractmethod
    def update_visualization(self, history_sample, planned_sample, t):
        """
        :param history_sample: sample containing where agent has been
        :param planned_sample: sample containing where agent plans to go
        :param t: current time step
        """
        raise NotImplementedError('Implement in subclass')

    @abc.abstractmethod
    def get_image(self, sample):
        """
        :param sample: rll_quadrotor Sample
        :return: numpy array or None
        """
        raise NotImplementedError('Implement in subclass')
