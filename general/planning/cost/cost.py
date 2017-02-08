import abc
from general.state_info.sample import Sample
from general.planning.cost.approx import CostApprox

class Cost(object):
    """Cost superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def eval(self, sample):
        """
        Evaluate this Cost with respect to the given sample

        :type sample: Sample
        :rtype: CostApprox
        """
        raise NotImplementedError("Must be implemented in subclass")

    def eval_vec(self, x, u):
        sample = Sample(1)
        sample.set_X(x, t=0)
        sample.set_U(u, t=0)
        cst = self.eval(sample)
        return cst