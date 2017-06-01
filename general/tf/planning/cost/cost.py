import abc
from general.state_info.sample import Sample

class Cost(object):
    """Cost superclass
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    def eval(self, data):
        """
        Evaluate this Cost with respect to the data
        :type data: tf tensor
        :rtype: tf tensor
        """
        raise NotImplementedError("Must be implemented in subclass")
