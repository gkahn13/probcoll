import abc
import os
from general.utility.logger import get_logger
from config import params, load_params

class Train:

    def __init__(self, data_dirs=None, plot_dir=None, add_data=False, asynch=False):
        if data_dirs is None:
            self._data_dirs = [os.path.join(params['exp_dir'], params['exp_name'])]
        else:
            self._data_dirs = data_dirs
        self._plot_dir = plot_dir
        self.asynch = asynch
        self.add_data = add_data
        self._logger = get_logger(
            self.__class__.__name__,
            params['model']['logger'])
        self._setup()

    @abc.abstractmethod
    def _setup(self):
        self._probcoll_model = None

    def _itr_dir(self, itr, data_dir):
        assert(type(itr) is int)
        dir = os.path.join(data_dir, 'itr{0}'.format(itr))
        return dir
    
    def _itr_samples_file(self, itr, data_dir):
        fname = os.path.join(self._itr_dir(itr, data_dir), 'samples_itr_{0}.npz'.format(itr))
        return fname
    
    def _get_files(self):
        files = []
        for data_dir in self._data_dirs:
            itr = 0
            while True:
                f = self._itr_samples_file(itr, data_dir)
                if os.path.exists(f):
                    files.append(f)
                else:
                    break
                itr += 1
        return files

    def run(self):
        if self.asynch:
            self._probcoll_model.train_loop()
        else:
            try:
                if self.add_data:
                    self._probcoll_model.add_data(self._get_files())
                self._probcoll_model.train()
                self._probcoll.run_testing(0)
                self._analyze.run_testing()
            finally:
                self._probcoll.close()
