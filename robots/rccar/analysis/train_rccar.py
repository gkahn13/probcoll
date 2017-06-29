from general.analysis.train import Train
from robots.rccar.algorithm.probcoll_rccar import ProbcollRCcar
from robots.rccar.algorithm.probcoll_model_rccar import ProbcollModelRCcar
from robots.rccar.analysis.analyze_rccar import AnalyzeRCcar

class TrainRCcar(Train):
    def _setup(self):
        if self.asynch:
            self._probcoll_model = ProbcollModelRCcar(save_dir=self._plot_dir)
        else:
            if self.add_data:
                self._probcoll= ProbcollRCcar(save_dir=self._plot_dir)
            else:
                self._probcoll= ProbcollRCcar(save_dir=self._plot_dir, data_dir=self._data_dirs[0])
            self._probcoll_model = self._probcoll.probcoll_model
            self._analyze = AnalyzeRCcar()
