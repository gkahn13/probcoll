from general.algorithm.train import Train
from robots.rccar.algorithm.probcoll_rccar import ProbcollRCcar
from robots.rccar.algorithm.probcoll_model_rccar import ProbcollModelRCcar

class TrainRCcar(Train):
    def _setup(self):
        if self.asynch:
            self._probcoll_model = ProbcollModelRCcar(save_dir=self._plot_dir)
        else:
            self._probcoll= ProbcollRCcar(save_dir=self._plot_dir)
            self._probcoll_model = self._probcoll.probcoll_model
