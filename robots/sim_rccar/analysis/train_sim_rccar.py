from general.analysis.train import Train
from general.algorithm.probcoll_model import ProbcollModel
from robots.sim_rccar.algorithm.probcoll_sim_rccar import ProbcollSimRCcar
from robots.sim_rccar.analysis.analyze_sim_rccar import AnalyzeSimRCcar

class TrainSimRCcar(Train):
    def _setup(self):
        if self.asynch:
            self._probcoll_model = ProbcollModel(save_dir=self._plot_dir)
        else:
            if self.add_data:
                self._probcoll= ProbcollSimRCcar(save_dir=self._plot_dir)
            else:
                self._probcoll= ProbcollSimRCcar(save_dir=self._plot_dir, data_dir=self._data_dirs[0])
            self._probcoll_model = self._probcoll.probcoll_model
            self._analyze = AnalyzeSimRCcar(save_dir=self._plot_dir)
