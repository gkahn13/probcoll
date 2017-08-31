import abc
import os, pickle

from general.utility.logger import get_logger
from general.state_info.sample import Sample

from config import params, load_params

class Analyze:

    def __init__(self, on_replay=False, save_dir=None):
        if save_dir is None:
            self._save_dir = os.path.join(params['exp_dir'], params['exp_name'])
        else:
            self._save_dir = save_dir

        yaml_path = self._get_yaml(self._save_dir)
        load_params(yaml_path)
        params['yaml_path'] = yaml_path
        
        self.on_replay = on_replay
        if self.on_replay:
            self._save_dir = os.path.join(self._save_dir, 'replay')

        self._logger = get_logger(self.__class__.__name__, 'info')

    #############
    ### Files ###
    #############

    def _get_yaml(self, dir_name):
        yamls = [fname for fname in os.listdir(dir_name) if '.yaml' in fname and '~' not in fname]
        yaml_path = os.path.join(dir_name, yamls[0])
        return yaml_path
    
    def _samples_dir(self):
        return os.path.join(self._save_dir, "samples")
    
    @property
    def _image_folder(self):
        path = os.path.join(self._save_dir, 'analysis_images')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @property
    def _plot_stats_file(self):
        return os.path.join(self._save_dir, self._image_folder, 'stats_{0}.png'.format(params['exp_name']))

    @property
    def _plot_stats_file_pkl(self):
        return self._plot_stats_file.replace('.png', '.pkl')

    @property
    def _plot_testing_stats_file(self):
        return os.path.join(self._save_dir, self._image_folder, 'testing_stats_{0}.png'.format(params['exp_name']))

    @property
    def _plot_testing_stats_file_pkl(self):
        return self._plot_testing_stats_file.replace('.png', '.pkl')

    @property
    def _plot_pred_mean_file(self):
        return os.path.join(self._save_dir, self._image_folder, 'pred_mean_{0}.png'.format(params['exp_name']))

    def _plot_get_stats_file(self, prefix='', testing=False):
        if testing:
            prefix = prefix + '_testing'
        return os.path.join(self._save_dir, self._image_folder, '{0}_stats_{1}.png'.format(prefix, params['exp_name']))

    @property
    def _plot_pred_std_file(self):
        return os.path.join(self._save_dir, self._image_folder, 'pred_std_{0}.png'.format(params['exp_name']))

    @property
    def _plot_pred_cost_file(self):
        return os.path.join(self._save_dir, self._image_folder, 'pred_cost_{0}.png'.format(params['exp_name']))

    @property
    def _plot_pred_std_hist_file(self):
        return os.path.join(self._save_dir, self._image_folder, 'pred_std_hist_{0}.png'.format(params['exp_name']))

    def _plot_mean_samples_prediction_file(self, itr):
        return os.path.join(self._save_dir, self._image_folder, 'mean_samples_prediction_itr{0}_{0}.png'.format(itr, params['exp_name']))

    def _plot_std_samples_prediction_file(self, itr):
        return os.path.join(self._save_dir, self._image_folder, 'std_samples_prediction_itr{0}_{0}.png'.format(itr, params['exp_name']))

    @property
    def _plot_samples_file(self):
        return os.path.join(self._save_dir, self._image_folder, 'samples_{0}.png'.format(params['exp_name']))

    @property
    def _plot_pred_groundtruth_file(self):
        return os.path.join(self._save_dir, self._image_folder, 'pred_groundtruth_{0}.png'.format(params['exp_name']))

    def _itr_load_samples(self, itr):
        fname = os.path.join(self._samples_dir(), 'samples_itr_{0}.npz'.format(itr))
        return Sample.load_with_time(fname)

    def _itr_load_testing_samples(self, itr):
        fname = os.path.join(self._samples_dir(), 'testing_samples_itr_{0}.npz'.format(itr))
        if os.path.exists(fname):
            return Sample.load_with_time(fname)
        else:
            return None, None

    #######################
    ### Data processing ###
    #######################

    def _load_samples(self):
        samples_itrs = []
        times = []
        files = os.listdir(self._samples_dir())
        sorted_files = []
        for f in files:
            if f[:7] == 'samples':
                sorted_files.append(f)
        sorted_files = sorted(sorted_files, key=lambda x: int(x[12:-4]))
        for itr, f in enumerate(sorted_files):
                fname = os.path.join(self._samples_dir(), f)
                samples, time = Sample.load_with_time(fname)
                samples_itrs.append((itr, samples))
                times.append(time)
        self._logger.info('Loaded {0} iteration of samples'.format(len(samples_itrs)))
        return samples_itrs, times

    def _load_testing_samples(self):
        max_itr = 0
        samples_itrs = []
        times = []
        files = os.listdir(self._samples_dir())
        sorted_files = []
        for f in files:
            if f[:7] == 'testing':
                sorted_files.append(f)
        sorted_files = sorted(sorted_files, key=lambda x: int(x[20:-4]))
        for itr, f in enumerate(sorted_files):
                fname = os.path.join(self._samples_dir(), f)
                samples, time = Sample.load_with_time(fname)
                samples_itrs.append((itr, samples))
                times.append(time)
        self._logger.info('Loaded {0} testing iteration of samples'.format(len(samples_itrs)))
        return samples_itrs, times
    
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

    def run_testing(self):
        raise NotImplementedError('Implement in subclass')
