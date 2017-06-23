import abc
import os
from general.utility.logger import get_logger
from config import params, load_params

class Train:

    def __init__(self, plot_dir, asynch=False):
        self._data_dir = os.path.join(params['exp_dir'], params['exp_name'])
        yamls = [fname for fname in os.listdir(self._data_dir) if '.yaml' in fname and '~' not in fname]
        yaml_path = os.path.join(self._data_dir, yamls[0])
        params['yaml_path'] = yaml_path
        self._plot_dir = plot_dir
        self.asynch = asynch
        self._logger = get_logger(
            self.__class__.__name__,
            params['model']['logger'])
        self._setup()

    @abc.abstractmethod
    def _setup(self):
        self._probcoll_model = None

    def _itr_dir(self, itr):
        assert(type(itr) is int)
        dir = os.path.join(self._data_dir, 'itr{0}'.format(itr))
        return dir
    
    def _itr_samples_file(self, itr):
        fname = os.path.join(self._itr_dir(itr), 'samples_itr_{0}.npz'.format(itr))
        return fname
    
    def _get_files(self):
        files = []
        itr = 0
        while True:
            f = self._itr_samples_file(itr)
            if os.path.exists(f):
                files.append(f)
            else:
                break
            itr += 1
        return files

    def run(self):
        if self.asynch:
            self._probcoll_model.async_train_func()
        else:
            self._probcoll_model.add_data(self._get_files())
            self._probcoll_model.train()
