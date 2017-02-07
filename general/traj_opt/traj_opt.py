import abc

class Trajopt(object):

    def __init__(self, dynamics, world, agent):
        """
        :type dynamics: Dynamics
        :type world: World
        :type agent: Agent
        """
        self.dynamics = dynamics
        self.world = world
        self.agent = agent

    @abc.abstractmethod
    def plan(self, x0, additional_costs=[]):
        """
        :param x0: start state
        :param additional_costs: extra iLQR costs (e.g. CostKL)
        :type additional_costs: rll_quadrotor Cost
        :return: Sample, LinearGaussianPolicy
        """
        raise NotImplementedError('Implement in subclass')
