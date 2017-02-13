import abc
import os, pickle

from general.utility.logger import get_logger
from general.state_info.sample import Sample

from config import params, load_params

class Analyze:

    def __init__(self, on_replay=False, parent_exp_dir=None):
        self.on_replay = on_replay
        self._save_dir = os.path.join(params['exp_dir'], params['exp_name'])

        yamls = [fname for fname in os.listdir(self._save_dir) if '.yaml' in fname and '~' not in fname]
        assert(len(yamls) == 1)
        yaml_path = os.path.join(self._save_dir, yamls[0])
        load_params(yaml_path)
        params['yaml_path'] = yaml_path

        if self.on_replay:
            self._save_dir = os.path.join(self._save_dir, 'replay')

        self._logger = get_logger(self.__class__.__name__, 'info')

    #############
    ### Files ###
    #############

    def _itr_dir(self, itr):
        assert(type(itr) is int)
        dir = os.path.join(self._save_dir, 'itr{0}'.format(itr))
        return dir

    def _itr_model_file(self, itr):
        return os.path.join(self._itr_dir(itr), 'model_itr_{0}.ckpt'.format(itr)).replace('replay/', '')

    @property
    def _image_folder(self):
        path = os.path.join(self._save_dir, 'images')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @property
    def _plot_stats_file(self):
        return os.path.join(self._save_dir, self._image_folder, 'stats_{0}.png'.format(params['exp_folder']))

    @property
    def _plot_stats_file_pkl(self):
        return self._plot_stats_file.replace('.png', '.pkl')

    @property
    def _plot_pred_mean_file(self):
        return os.path.join(self._save_dir, self._image_folder, 'pred_mean_{0}.png'.format(params['exp_folder']))

    @property
    def _plot_pred_std_file(self):
        return os.path.join(self._save_dir, self._image_folder, 'pred_std_{0}.png'.format(params['exp_folder']))

    @property
    def _plot_pred_cost_file(self):
        return os.path.join(self._save_dir, self._image_folder, 'pred_cost_{0}.png'.format(params['exp_folder']))

    @property
    def _plot_pred_std_hist_file(self):
        return os.path.join(self._save_dir, self._image_folder, 'pred_std_hist_{0}.png'.format(params['exp_folder']))

    def _plot_mean_samples_prediction_file(self, itr):
        return os.path.join(self._save_dir, self._image_folder, 'mean_samples_prediction_itr{0}_{0}.png'.format(itr, params['exp_folder']))

    def _plot_std_samples_prediction_file(self, itr):
        return os.path.join(self._save_dir, self._image_folder, 'std_samples_prediction_itr{0}_{0}.png'.format(itr, params['exp_folder']))

    @property
    def _plot_samples_file(self):
        return os.path.join(self._save_dir, self._image_folder, 'samples_{0}.png'.format(params['exp_folder']))

    @property
    def _plot_pred_groundtruth_file(self):
        return os.path.join(self._save_dir, self._image_folder, 'pred_groundtruth_{0}.png'.format(params['exp_folder']))

    def _itr_load_samples(self, itr):
        fname = os.path.join(self._itr_dir(itr), 'samples_itr_{0}.npz'.format(itr))
        return Sample.load(fname)

    def _itr_load_mpcs(self, itr):
        fname = os.path.join(self._itr_dir(itr), 'mpcs_itr_{0}.pkl'.format(itr))
        with open(fname, 'r') as f:
            d = pickle.load(f)
        return d

    def _itr_load_worlds(self, itr):
        fname = os.path.join(self._itr_dir(itr), 'worlds_itr_{0}.pkl'.format(itr))
        with open(fname, 'r') as f:
            d = pickle.load(f)
        return d

    #######################
    ### Data processing ###
    #######################

    def _load_samples(self):
        samples_itrs = []

        itr = 0
        while True:
            try:
                samples_itrs.append(self._itr_load_samples(itr))
                itr += 1
            except:
                break

        self._logger.info('Loaded {0} iterations of samples'.format(len(samples_itrs)))

        ### load initial dataset
        init_data_folder = params['probcoll'].get('init_data', None)
        if init_data_folder is not None:
            if itr == 0:
                samples_itrs.append([])
            num_init_samples = 0

            fnames = [os.path.join(init_data_folder, fname) for fname in os.listdir(init_data_folder)]
            for fname in fnames:
                try:
                    samples = Sample.load(fname)
                except:
                    continue

                self._logger.debug('Loaded samples from {0}'.format(fname))
                samples_itrs[0] += samples
                num_init_samples += len(samples)

            self._logger.info('Loaded initial dataset of {0} samples'.format(num_init_samples))

        return samples_itrs

    def _load_mpcs(self):
        mpcs_itrs = []

        itr = 0
        while True:
            try:
                mpcs_itrs.append(self._itr_load_mpcs(itr))
                itr += 1
            except:
                break

        return mpcs_itrs

    def _load_worlds(self):
        worlds_itrs = []

        itr = 0
        while True:
            try:
                worlds_itrs.append(self._itr_load_worlds(itr))
                itr += 1
            except:
                break

        return worlds_itrs

    ################
    ### Plotting ###
    ################

    ### Implement in subclass

    ###########
    ### Run ###
    ###########

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError('Implement in subclass')