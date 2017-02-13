import os
import re
import pickle, yaml, shutil
import numpy as np
import scipy, scipy.misc
import h5py

from general.utility.logger import get_logger

from config import params, path

# make sure FileManager conforms to this pattern
_FILE_PATTERN = re.compile(
    r"(?P<pdir>.*)/exp(?P<exp>[0-9]+)/iter(?P<itr>[0-9]+)/" +
    r"(?P<sdir>\w+)/sample_cond(?P<cond>[0-9]+)_rep(?P<rep>[0-9]+)\.h5"
)

# all directories under iterX
_ITER_SUBDIRS = [
    'nn_before', # nn policy rollout before training
    'nn_test',  # nn policy rollout for test condition
    'nn_train',   # nn policy rollout for training condition
    'test',   # iLQR offline optimized trajectories
    'train',  # processed rollout of MPC/LQR for training nn
    'model',  # nn related files: <nn>model, scalers, invSigma
    'sim',  # all simulation rollout of iLQR/MPC
    'curriculum', # crashes from curriculum learning
]

# all directories under expX, except iterY
_EXP_SUBDIRS = [
    'tmp',  # temp folder for read-only access
]


class FileManager(object):
    """
    Directory structure

    parent_exp_dir/
      exp0/
        iter0/
          train/
            list.txt
            sample_condX_repY.h5
            ...
          test/
            list.txt
            sample_condX_repY.h5
          nn_train/
            list.txt
            sample_condX_repY.h5
          model/
            _iter_Z.caffemodel
        iter1/
          ...
        ...
    """
    # __metaclass__ = FixedSingleton

    def __init__(self, exp_num=None, exp_folder=None, read_only=False, continue_writing=False, parent_exp_dir=None):
        """
        :param exp_num: experiment number. if None, will automatically find next number in folder
        :param exp_folder: explicit path to folder with exp_dir (will then ignore exp_num)
        :param read_only: True allows for post-data processing (i.e. getting file paths)
        :param continue_writing: can continue writing to this folder
        """
        self._logger = get_logger(self.__class__.__name__, 'debug')
        self._read_only = read_only
        self._filename_pattern = _FILE_PATTERN
        if parent_exp_dir is None:
            self.parent_exp_dir = os.path.join(
                                    os.path.join(path, '../experiments'),
                                    params['exp_dir'])
        else:
            self.parent_exp_dir = os.path.join(parent_exp_dir, params['exp_dir'])
        self._logger.info("FileManager: using %s as parent folder" % self.parent_exp_dir)

        if exp_folder is not None:
            dir_path = os.path.abspath(os.path.join(self.parent_exp_dir, exp_folder))
        else:
            # find proper exp_num and error checking
            exp_folder_nums = [
                int(d.replace('exp', ''))
                for d in os.listdir(self.parent_exp_dir)
                if os.path.isdir(os.path.join(self.parent_exp_dir, d)) and ('exp' in d)
            ]
            if exp_num is None:
                exp_num = max(exp_folder_nums) + 1 if len(exp_folder_nums) > 0 else 0

            # resolve dir (_get_dir only creates if not read-only)
            dir_path = os.path.abspath(os.path.join(self.parent_exp_dir, 'exp{0}'.format(exp_num)))

        dir_exists = os.path.exists(dir_path)
        # if not dir_exists and continue_writing:
        #     raise ValueError('Cannot continue writing since folder does not exist')
        if dir_exists and read_only is False and not continue_writing:
            raise ValueError('attempted over-write existing exp')
        if not dir_exists and read_only is True:
            raise ValueError('read-only exp%d dir not found', exp_num)

        self._dir = self._get_dir(dir_path)
        self._logger.info("FileManager: exp folder as %s" % self._dir)

        self._params_path = os.path.join(self._dir, 'params.pickle')
        self._debug_log_path = os.path.join(self._dir, 'debug_log.txt')
        self._dir_tmp = os.path.join(self._dir, 'tmp')
        if not os.path.exists(self._dir_tmp): os.makedirs(self._dir_tmp)

        if not read_only:
            shutil.copy(params['yaml_path'],
                        os.path.join(self._dir, os.path.basename(params['yaml_path'])))

    """ Internal Methods """

    def _get_dir(self, d):
        assert not os.path.isfile(d), 'attempted to overwrite file or dir'
        if not os.path.isdir(d):
            if not self._read_only:
                os.makedirs(d)
            else:
                print 'attempted access to non-existing dir %s' % d
                print 'redirect to tmp/ folder'
                d = self._dir_tmp
        return d

    def _iter_dir_path(self, itr):
        return os.path.join(self._dir, 'iter%d' % itr)

    def _find_dir(self, sub_dir=None, itr=None):
        if sub_dir in _EXP_SUBDIRS:
            assert itr is None
            dir_path = self._get_dir(os.path.join(self._dir, sub_dir))
        elif sub_dir in _ITER_SUBDIRS:
            assert isinstance(itr, int) and itr >= 0
            dir_path = self._get_dir(os.path.join(
                self._get_dir(os.path.join(self._dir, 'iter%d' % itr)),
                sub_dir))
        elif sub_dir is None and itr is None:
            dir_path = self._dir
        elif sub_dir is None and itr is not None:
            assert isinstance(itr, int) and itr >= 0
            dir_path = self._get_dir(self._iter_dir_path(itr))
        else:
            raise ValueError('Unknown input sub_dir=%s, itr=%s' % (sub_dir, itr))
        return dir_path

    def _list_path(self, sub_dir, itr=None):
        assert sub_dir is not None
        dir_path = self._find_dir(sub_dir, itr)
        path = os.path.join(dir_path, 'list.txt')
        return path

    def _sample_path(self, sub_dir, cond, rep=0, itr=None):
        assert sub_dir is not None
        # If this file name pattern is changed, update self._filename_re
        fn = 'sample_cond%d_rep%d.h5' % (cond, rep)
        dir_path = self._find_dir(sub_dir, itr)
        path = os.path.join(dir_path, fn)
        return path

    """ Interface methods for getting directory/file path """

    @property
    def dir(self):
        return self._dir

    @property
    def debug_log_path(self):
        return self._debug_log_path

    @property
    def params_path(self):
        return self._params_path

    @property
    def yaml_path(self):
        return self.params_path.replace('.pickle', '.yaml')
    
    def gps_dual_path(self, name, itr, cond):
        assert isinstance(itr, int) and itr >= 0
        fn = '%s_dual_cond%d.pkl' % (name, cond)
        path = os.path.join(self._find_dir('model', itr), fn)
        return path

    def curriculum_image_path(self, itr, cond):
        fn = 'curriculum_%d.png' % cond
        return os.path.join(self._find_dir('curriculum', itr), fn)

    # def nn_linearization(self, name, itr, cond):
    #     assert isinstance(itr, int) and itr >= 0
    #     fn = '%s_net_lr_cond%d.pkl' % (name, cond)
    #     path = os.path.join(self._find_dir('model', itr), fn)
    #     return path

    def nn_model_path(self, nn_module, name, itr, nn_itr=None):
        assert isinstance(itr, int) and itr >= 0
        suffix = '%smodel' % nn_module
        if nn_itr is None:
            model_dir = self._find_dir('model', itr)
            nn_itr = max([int(s.split('_')[-1].split('.')[0]) for s in os.listdir(model_dir) if suffix in s])
        fn = '%s_iter_%d.%s' % (name, nn_itr, suffix)
        path = os.path.join(self._find_dir('model', itr), fn)
        return path

    def scalers_path(self, name, itr):
        assert isinstance(itr, int) and itr >= 0
        fn = '%s_scalers.pkl' % name
        path = os.path.join(self._find_dir('model', itr), fn)
        return path

    def policy_var_path(self, name, itr):
        assert isinstance(itr, int) and itr >= 0
        fn = '%s_inv_sigma.pkl' % name
        path = os.path.join(self._find_dir('model', itr), fn)
        return path

    def training_data_path(self, itr):
        assert isinstance(itr, int) and itr >= 0
        fn = 'training_data_{0}_itr{1}.pkl'.format(os.path.basename(self._dir), itr)
        path = os.path.join(self._find_dir('train', itr), fn)
        return path

    """ Disk operations """

    def flush_scalers(self, name, scalers, itr):
        with open(self.scalers_path(name, itr), 'w') as f:
            pickle.dump(scalers, f)

    def load_scalers(self, name, itr):
        with open(self.scalers_path(name, itr), 'r') as f:
            scalers = pickle.load(f)
        return scalers

    def flush_policy_var(self, name, var, itr):
        with open(self.policy_var_path(name, itr), 'w') as f:
            pickle.dump(var, f)

    def load_policy_var(self, name, itr):
        with open(self.policy_var_path(name, itr), 'rb') as f:
            policy_var = pickle.load(f)
        return policy_var


    # def flush_dual(self, dual, cond, itr):
    #     with open(self.gps_dual_path(itr, cond), 'w') as f:
    #         pickle.dump(dual, f)
    #
    # def flush_lr(self, dct, cond, itr):
    #     with open(self.nn_linearization(itr, cond), 'w') as f:
    #         pickle.dump(dct, f)

    # def flush_training_data(self, itr, samples, Quus):
    #     X = np.vstack([s.get_X() for s in samples])
    #     U = np.vstack([s.get_U() for s in samples])
    #     O = np.vstack([s.get_O() for s in samples])
    #     Quus = np.vstack([Quu for Quu in Quus])
    #
    #     d = {'X': X, 'U': U, 'O': O, 'Quus': Quus}
    #
    #     with open(self.training_data_path(itr), 'w') as f:
    #         pickle.dump(d, f)
    #
    # def load_training_data(self, itrs=None):
    #     if itrs is None:
    #         itrs = []
    #         i = 0
    #         while os.path.exists(self._iter_dir_path(i)):
    #             itrs.append(i)
    #     elif isinstance(itrs, list):
    #         for i in itrs:
    #             assert(os.path.exists(self._iter_dir_path(i)))
    #     else:
    #         raise Exception('Invalid itr argument: {0}'.format(itrs))
    #
    #     X, U, O, Quus = [], [], [], []
    #     for i in itrs:
    #         fname = self.training_data_path(i)
    #         if os.path.exists(fname):
    #             with open(fname, 'r') as f:
    #                 d = pickle.load(f)
    #                 X.append(d['X'])
    #                 U.append(d['U'])
    #                 O.append(d['O'])
    #                 Quus.append(d['Quus'])
    #
    #     X = np.vstack(X)
    #     U = np.vstack(U)
    #     O = np.vstack(O)
    #     Quus = np.vstack(Quus)
    #
    #     return X, U, O, Quus

    def flush_training_data(self, itr, samples, Quus):
        d = {'samples': samples, 'Quus': Quus}

        with open(self.training_data_path(itr), 'w') as f:
            pickle.dump(d, f)

    def load_training_data(self, itrs=None):
        if itrs is None:
            itrs = []
            i = 0
            while os.path.exists(self._iter_dir_path(i)):
                itrs.append(i)
        elif isinstance(itrs, list):
            for i in itrs:
                assert(os.path.exists(self._iter_dir_path(i)))
        else:
            raise Exception('Invalid itr argument: {0}'.format(itrs))

        X, U, O, Quus = [], [], [], []
        for i in itrs:
            fname = self.training_data_path(i)
            if os.path.exists(fname):
                with open(fname, 'r') as f:
                    d = pickle.load(f)
                    samples = d['samples']
                    for sample in samples:
                        X.append(sample.get_X())
                        U.append(sample.get_U())
                        O.append(sample.get_O())
                    Quus += d['Quus']

        X = np.vstack(X)
        U = np.vstack(U)
        O = np.vstack(O)
        Quus = np.vstack(Quus)

        return X, U, O, Quus

    # TODO_TZ: make sure this works with CGT too
    def flush_samples(self, sub_dir, samples, cond, itr=None, prec=None,
                      no_h5=False, images=None, crashes=None):
        list_file = self._list_path(sub_dir, itr)
        if not isinstance(samples, list):
            samples = [samples]
        for rep, s in enumerate(samples):
            fn = self._sample_path(sub_dir, cond, rep, itr)
            if crashes is not None: crash = crashes[rep]
            else: crash = False

            if crash: fn += '_crash'
            if not no_h5:
                # with 'w-' flag, error if file already exists
                with h5py.File(fn, 'w') as f:
                    # the dimension for HDF5 must be
                    # num_examples * num_channels * width * height
                    f['states'] = mat_to_caffe(s.get_X())
                    f['obs'] = mat_to_caffe(s.get_O())
                    f['control'] = s.get_U()
                    if prec is not None:
                        f['prec'] = prec
            if images is not None and images[rep] is not None:
                scipy.misc.imsave(fn+'.png', images[rep])
            if not crash and not no_h5:
                with open(list_file, 'a') as f:
                    f.write("%s\n" % (fn,))

    # TODO_TZ: make sure this works with CGT too
    def load_samples(self, sub_dir, itr):
        itr_dir = self._find_dir(sub_dir, itr)
        h5_files = filter(lambda f: f.endswith('.h5'), os.listdir(itr_dir))
        h5_files = [os.path.join(itr_dir, f) for f in h5_files]

        x, obs, tgt_mu, prec = [], [], [], []
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                x.append(caffe_to_mat(f['states']))
                obs.append(caffe_to_mat(f['obs']))
                tgt_mu.append(caffe_to_mat(f['control']))
                if 'prec' in f.keys():
                    prec.append(caffe_to_mat(f['prec']))
        return np.array(x), np.array(obs), np.array(tgt_mu), np.array(prec)

    def flush_images(self, itr, cond, rep, images):
        dir = self._find_dir('train', itr)
        for i, im in enumerate(images):
            fname = os.path.join(dir, 'cond{0}_rep{1}_im{2:04d}.png'.format(cond, rep, i))
            scipy.misc.imsave(fname, im)

    """ High-level operations """

    # @staticmethod
    # def probe(list_path):
    #     """
    #     Returned a list of file paths as specified in the file list.
    #
    #     :rtype: list[str]
    #     """
    #     with open(list_path, 'r') as f:
    #         paths = [''.join(l.strip()) for l in f.readlines()]
    #     return paths

    # def split_validation(self, name, gps_iter, validate_conds, append=False, shuffle=True):
    #     # # checks
    #     sim_list = self.probe(self._list_path('train', gps_iter))
    #     assert len(sim_list) > 1, 'Require at least two conditions'
    #     # generate validation file list
    #     valid_list = []
    #     for file in sim_list:
    #         m = self._filename_pattern.match(file)
    #         if int(m.group('cond')) in validate_conds:
    #             valid_list.append(file)
    #     # generate training file list
    #     train_list = []
    #     for file in sim_list:
    #         m = self._filename_pattern.match(file)
    #         if int(m.group('cond')) not in validate_conds:
    #             train_list.append(file)
    #     # shuffle
    #     if shuffle:
    #         random.shuffle(valid_list)
    #         random.shuffle(train_list)
    #     # write to file
    #     write = 'a' if append else 'w'
    #     with open(self.caffe_test_list_path(name), write) as f:
    #         f.write('\n'.join(valid_list)+'\n')
    #     with open(self.caffe_train_list_path(name), write) as f:
    #         f.write('\n'.join(train_list)+'\n')
    #     return valid_list, train_list

    # def split_validation_multiple(self, name, gps_iter_list, validate_conds_list, shuffle=True):
    #     valid_list = []
    #     train_list = []
    #     for gps_iter, validate_conds in zip(gps_iter_list, validate_conds_list):
    #         # # checks
    #         sim_list = self.probe(self._list_path('train', gps_iter))
    #         assert len(sim_list) > 1, 'Require at least two conditions'
    #         # generate validation file list
    #         for file in sim_list:
    #             m = self._filename_pattern.match(file)
    #             if int(m.group('cond')) in validate_conds:
    #                 valid_list.append(file)
    #         # generate training file list
    #         for file in sim_list:
    #             m = self._filename_pattern.match(file)
    #             if int(m.group('cond')) not in validate_conds:
    #                 train_list.append(file)
    #     # shuffle
    #     if shuffle:
    #         random.shuffle(valid_list)
    #         random.shuffle(train_list)
    #     # write to file
    #     with open(self.caffe_test_list_path(name), 'w') as f:
    #         f.write('\n'.join(valid_list)+'\n')
    #     with open(self.caffe_train_list_path(name), 'w') as f:
    #         f.write('\n'.join(train_list)+'\n')
    #     return valid_list, train_list

    # def create_scalers(self, train_files,
    #                    has_x=True, has_o=True, has_u=True):
    #     """
    #     Create sklearn.preprocessing.StandardScaler which scales (mean shift
    #     and variance removal) the HDF5 files for iter-th GPS iteration.
    #
    #     :rtype: (StandardScaler, StandardScaler)
    #     """
    #     scalers = {}
    #     all_X, all_U, all_O = [], [], []
    #     for file_path in train_files:
    #         with h5py.File(file_path, 'r') as f:
    #             if has_o: all_O.append(caffe_to_mat(f['obs']))
    #             if has_x: all_X.append(caffe_to_mat(f['states']))
    #             if has_u: all_U.append(caffe_to_mat(f['control']))
    #     print "HDF5Writer: creating scalers from %d h5 files" % (len(all_X))
    #     if has_u:
    #         U_all = np.concatenate(all_U, axis=0)
    #         scaler_U = StandardScaler().fit(U_all)
    #         scalers['u'] = scaler_U
    #         print "HDF5Writer: U with mean %s, \n\tstd %s" % \
    #               (str(scaler_U.mean_), str(scaler_U.std_))
    #     if has_x:
    #         X_all = np.concatenate(all_X, axis=0)
    #         scaler_X = StandardScaler().fit(X_all)
    #         scalers['x'] = scaler_X
    #         print "HDF5Writer: X with mean %s, \n\tstd %s" % \
    #               (str(scaler_X.mean_), str(scaler_X.std_))
    #     if has_o:
    #         O_all = np.concatenate(all_O, axis=0)
    #         scaler_O = StandardScaler().fit(O_all)
    #         scalers['o'] = scaler_O
    #         print "HDF5Writer: O with mean %s, \n\tstd %s" % \
    #               (str(scaler_O.mean_), str(scaler_O.std_))
    #     return scalers

    # def scale_dataset(self, files, scalers):
    #     """
    #     Scale (mean shift and variance removal) the HDF5 files specifed in the
    #     file list. If scaler is provided, scale the dataset by it. Otherwise,
    #     create new scalers for states and control. The scalers are returned.
    #     """
    #     for file in files:
    #         # with r+ flag, file must exist
    #         with h5py.File(file, 'r+') as f:
    #             if 'u' in scalers:
    #                 U = scalers['u'].transform(caffe_to_mat(f['control']))
    #                 if 's_control' in f: del f['s_control']
    #                 f['s_control'] = U
    #             if 'x' in scalers:
    #                 X = scalers['x'].transform(caffe_to_mat(f['states']))
    #                 if 's_states' in f: del f['s_states']
    #                 f['s_states'] = mat_to_caffe(X)
    #             if 'o' in scalers:
    #                 O = scalers['o'].transform(caffe_to_mat(f['obs']))
    #                 if 's_obs' in f: del f['s_obs']
    #                 f['s_obs'] = mat_to_caffe(O)

    def __getstate__(self):
        # pass
        return dict()

    def __setstate__(self, state):
        # FileManager is a singleton
        # if an instance exists before unpickle
        # when pickle calls FileManager.__new__, the pre-existing instance
        # will be returned and reinstated as self in this current scope
        pass


def mat_to_caffe(x):
    """
    Convert (dX,) or (N, dX) array to (N=1, 1, 1, dX) ignore (N, 1, 1, dX)
    """
    if not isinstance(x, np.ndarray): x = np.array(x, dtype=np.float64)
    GPS_ASSERT(x.ndim <= 4)
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
    if x.ndim == 2:
        x = np.expand_dims(np.expand_dims(x, axis=1), axis=1)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=1)
    # GPS_SHAPE(x, (False, 1, 1, False))
    return x


def caffe_to_mat(x):
    """
    Convert (N, [1, 1,] dX) array to (N, dX) or (dX,)
    """
    if not isinstance(x, np.ndarray): x = np.array(x, dtype=np.float64)
    if x.ndim == 4:
        # GPS_SHAPE(x, (False, 1, 1, False))
        x = np.squeeze(x, axis=(1,2))
    # GPS_ASSERT(x.ndim < 3, 'dimension must be 1 or 2 or 4')
    if x.shape[0] == 1:
        x = np.squeeze(x, axis=0)
    return x
