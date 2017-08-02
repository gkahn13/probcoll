from general.analysis.train import Train
from general.algorithm.probcoll_model import ProbcollModel
from robots.bebop2d.algorithm.probcoll_bebop2d import ProbcollBebop2d

class TrainBebop2d(Train):
    def _setup(self):
        if self.asynch:
            self._probcoll_model = ProbcollBebop2d(save_dir=self._plot_dir)
        else:
            if self.add_data:
                self._probcoll= ProbcollBebop2d(save_dir=self._plot_dir)
            else:
                self._probcoll= ProbcollBebop2d(save_dir=self._plot_dir, data_dir=self._data_dirs[0])
            self._probcoll_model = self._probcoll.probcoll_model
