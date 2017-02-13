from general.algorithm.cost_probcoll import CostProbcoll


class CostProbcollBebop2d(CostProbcoll):

    def __init__(self, bootstrap, **kwargs):
        CostProbcoll.__init__(self, bootstrap, **kwargs)
