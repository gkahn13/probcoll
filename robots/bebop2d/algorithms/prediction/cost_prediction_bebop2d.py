from general.algorithms.prediction.cost_prediction import CostPrediction


class CostPredictionBebop2d(CostPrediction):

    def __init__(self, bootstrap, **kwargs):
        CostPrediction.__init__(self, bootstrap, **kwargs)
