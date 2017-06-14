import abc

class Planner(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, H, dynamics, cost_funcs, use_mpc):
        self._H = H
        self._dynamics = dynamics
        self._cost_funcs = cost_funcs
        self._use_mpc = use_mpc

    @abc.abstractmethod
    def plan(self, x, o):
        raise NotImplementedError('Implement in subclass')

    @abc.abstractmethod
    def _mpc_update(self):
        raise NotImplementedError('Implement in sublcass')