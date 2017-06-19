from general.algorithm.train import Train
from robots.rccar.algorithm.probcoll_model_rccar import ProbcollModelRCcar

class TrainRCcar(Train):
    def _setup(self):
        self._probcoll_model = ProbcollModelRCcar(save_dir=self._plot_dir)
