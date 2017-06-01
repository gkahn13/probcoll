import abc

class Planner(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def plan(self, x, o):
        raise NotImplementedError('Implement in subclass')
