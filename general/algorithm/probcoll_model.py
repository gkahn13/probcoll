import abc

import os, pickle
import random, time
import itertools
import shutil
from collections import defaultdict
import hashlib

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from general.utility.file_manager import FileManager
from general.utility.utils import TimeIt

from general.utility.logger import get_logger
from general.state_info.sample import Sample

from config import params

class MLPlotter:
    """
    Plot/save machine learning data
    """
    def __init__(self, title, subplot_dicts, shape=None, figsize=None):
        """
        :param title: title of plot
        :param subplot_dicts: dictionary with dictionaries of form
                              name: {subplot, title, color, ylabel}
        """
        ### setup plot figure
        num_subplots = max(d['subplot'] for d in subplot_dicts.values()) + 1
        if shape is None:
            shape = (1, num_subplots)
        if figsize is None:
            figsize = (30, 7)
        self.f, self.axes = plt.subplots(shape[0], shape[1], figsize=figsize)
        mng = plt.get_current_fig_manager()
        mng.window.showMinimized()
        plt.suptitle(title)
        plt.show(block=False)
        plt.pause(0.5)

        self.train_lines = {}
        self.val_lines = {}

        axes = self.axes.ravel().tolist()
        for name, d in subplot_dicts.items():
            ax = axes[d['subplot']]
            ax.set_xlabel('Training samples')
            if 'title' in d: ax.set_title(d['title'])
            if 'ylabel' in d: ax.set_ylabel(d['ylabel'])

            self.train_lines[name] = ax.plot([], [], color=d['color'], linestyle='-', label=name)[0]
            self.val_lines[name] = ax.plot([], [], color=d['color'], linestyle='--')[0]

            ax.legend()

        self.f.canvas.draw()
        plt.pause(0.5)

    def _update_line(self, line, new_x, new_y):
        xdata, ydata = line.get_xdata(), line.get_ydata()

        xdata = np.concatenate((xdata, [new_x]))
        ydata = np.concatenate((ydata, [new_y]))

        line.set_xdata(xdata)
        line.set_ydata(ydata)

        ax = line.get_axes()
        ax.relim()
        ax.autoscale_view()

    def add_train(self, name, training_samples, value):
        self._update_line(self.train_lines[name], training_samples, value)

    def add_val(self, name, value):
        xdata = self.train_lines[name].get_xdata()
        self._update_line(self.val_lines[name], xdata[-1] if len(xdata) > 0 else 0, value)

    def plot(self):
        self.f.canvas.draw()
        plt.pause(0.01)

    def save(self, save_dir, name='training.png'):
        self.f.savefig(os.path.join(save_dir, name))
        with open(os.path.join(save_dir, 'plotter.pkl'), 'w') as f:
            pickle.dump(dict([(k, (v.get_xdata(), v.get_ydata())) for k, v in self.train_lines.items()] +
                             [(k, (v.get_xdata(), v.get_ydata())) for k, v in self.val_lines.items()]),
                        f)



    def close(self):
        plt.close(self.f)

class ProbcollModel:
    __metaclass__ = abc.ABCMeta

    ####################
    ### Initializing ###
    ####################

    def __init__(self, dist_eps, read_only=False, finalize=True):
        self.dist_eps = dist_eps

        self.fm = FileManager(exp_folder=params['exp_folder'], read_only=read_only, continue_writing=True)
        self.save_dir = os.path.join(self.fm.dir, 'prediction')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.logger = get_logger(self.__class__.__name__, 'debug', os.path.join(self.save_dir, 'debug.txt'))

        self.random_seed = params['random_seed']
        for k, v in params['prediction']['model'].items():
            setattr(self, k, v)

        self.dX = len(self.X_idxs())
        self.dU = len(self.U_idxs())
        self.dO = len(self.O_idxs())
        self.doutput = len(self.output_idxs())

        self.npz_fnames = []
        self.tfrecords_train_fnames = []
        self.tfrecords_val_fnames = []
        self.preprocess_fnames = []

        self.X_mean = np.zeros(self.dX)
        self.U_mean = np.zeros(self.dU)
        self.O_mean = np.zeros(self.dO)

        code_file_exists = os.path.exists(self._code_file)
        if code_file_exists:
            self.logger.info('Creating OLD graph')
        else:
            self.logger.info('Creating NEW graph')
            shutil.copyfile(self._this_file, self._code_file)
        self._graph_inference = self._get_old_graph_inference(graph_type=self.graph_type)

        self.tf_debug = {}
        self._graph_setup()

    #############
    ### Files ###
    #############

    @abc.abstractproperty
    @property
    def _this_file(self):
        raise NotImplementedError('Implement in subclass')

    @property
    def _code_file(self):
        return os.path.join(os.path.abspath(self.save_dir),
                            'probcoll_model_{0}.py'.format(os.path.basename(self.fm._dir)))

    @property
    def _hash(self):
        """ Anything that if changed, need to re-save the data """
        d = {}

        pm_params = params['prediction']['model']
        for key in ('T', 'num_bootstrap', 'val_pct', 'X_order', 'U_order', 'O_order', 'output_order', 'balance',
                    'aggregate_save_data', 'save_type'):
            d[key] = pm_params[key]

        for key in ('X', 'U', 'O'):
            d[key] = params[key]

        return hashlib.md5(str(d)).hexdigest()

    ############
    ### Data ###
    ############

    def X_idxs(self, p=None, without=[]):
        if p is None: p = params
        return list(itertools.chain(*[range(p['X'][ord]['idx'],
                                            p['X'][ord]['idx']+p['X'][ord]['dim'])
                                      for ord in self.X_order if ord not in without]))

    def U_idxs(self, p=None, without=[]):
        if p is None: p = params
        return list(itertools.chain(*[range(p['U'][ord]['idx'],
                                            p['U'][ord]['idx']+p['U'][ord]['dim'])
                                      for ord in self.U_order if ord not in without]))

    def O_idxs(self, p=None, without=[]):
        if p is None: p = params
        return list(itertools.chain(*[range(p['O'][ord]['idx'],
                                            p['O'][ord]['idx']+p['O'][ord]['dim'])
                                      for ord in self.O_order if ord not in without]))

    def output_idxs(self, p=None, without=[]):
        if p is None: p = params
        return list(itertools.chain(*[range(p['O'][ord]['idx'],
                                            p['O'][ord]['idx']+p['O'][ord]['dim'])
                                      for ord in self.output_order if ord not in without]))

    ############
    ### Data ###
    ############

    def _modify_sample(self, sample):
        """
        In case you want to pre-process the sample before adding it
        :return: Sample
        """
        return [sample]

    def _load_samples(self, npz_fnames):
        """
        :param npz_fnames: pkl file names containing Samples
        :return: start_idxs_by_sample, X_by_sample, U_by_sample, O_by_sample, output_by_sample
                 (start_idxs_by_sample only contains start indices)
        """
        start_idxs_by_sample = []
        X_by_sample = []
        U_by_sample = []
        O_by_sample = []
        output_by_sample = []

        random.shuffle(npz_fnames)
        for npz_fname in npz_fnames:
            ### load samples
            # self.logger.debug('\tOpening {0}'.format(npz_fname))

            samples = Sample.load(npz_fname)
            random.shuffle(samples)

            ### add to data
            for og_sample in samples:
                for sample in self._modify_sample(og_sample):
                    s_params = sample._meta_data
                    X = sample.get_X()[:, self.X_idxs(p=s_params)]
                    U = sample.get_U()[:, self.U_idxs(p=s_params)]
                    O = sample.get_O()[:, self.O_idxs(p=s_params)]
                    output = sample.get_O()[:, self.output_idxs(p=s_params)]

                    buffer_len = 1
                    if len(X) < 1 + buffer_len: # used to be self.T, but now we are extending
                        continue

                    X_input, U_input, O_input = self._create_input(X, U, O)
                    output = self._create_output(output)

                    if int(output[-1, 0]) == 1:
                        # if collision, extend collision by T-1 (and buffer)
                        X_input = np.vstack((X_input, np.tile([X_input[-1]], (self.T - 1 - buffer_len, 1))))
                        U_input = np.vstack((U_input, np.tile([U_input[-1]], (self.T - 1 - buffer_len, 1))))
                        output = np.vstack((output, np.tile([output[-1]], (self.T - 1 - buffer_len, 1))))
                        O_input = np.vstack((O_input, np.tile([O_input[-1]], (self.T - 1 - buffer_len, 1))))

                    for arr in (X_input, U_input, O_input, output):
                        assert(np.all(np.isfinite(arr)))

                    X_by_sample.append(X_input)
                    U_by_sample.append(U_input)
                    O_by_sample.append(O_input)
                    output_by_sample.append(output)

                    ### just the start indices
                    start_idxs_by_sample.append(range(0, len(X_by_sample[-1]) - self.T + 1))

        return start_idxs_by_sample, X_by_sample, U_by_sample, O_by_sample, output_by_sample

    def _balance_data(self, start_idxs_by_sample, X_by_sample, U_by_sample, O_by_sample, output_by_sample):
        """
        Default is no balancing, just split normally
        """
        num_val = max(1, int(len(start_idxs_by_sample) * self.val_pct))

        ### split by traj
        start_idxs_by_val_sample = start_idxs_by_sample[:num_val]
        start_idxs_by_train_sample = start_idxs_by_sample[num_val:]
        ### idxs to resample from (sample, start_idx)
        def resample_idxs(start_idxs_by_sample):
            return [(i, j) for i in xrange(len(start_idxs_by_sample))
                    for j in start_idxs_by_sample[i]]
        train_resample_idxs = resample_idxs(start_idxs_by_train_sample)
        val_resample_idxs = resample_idxs(start_idxs_by_val_sample)
        ### do resampling
        def resample(resample_idxs):
            num_samples = len(resample_idxs)

            ### [# train/val samples, # bootstrap, start idxs]
            bootstrap_start_idxs_by_sample = [[[] for _ in xrange(self.num_bootstrap)] for _ in xrange(num_samples)]
            for b in xrange(self.num_bootstrap):
                for _ in xrange(num_samples):
                    sample_idx, start_idx = random.choice(resample_idxs)
                    bootstrap_start_idxs_by_sample[sample_idx][b].append(start_idx)

            return bootstrap_start_idxs_by_sample

        bootstrap_start_idxs_by_train_sample = resample(train_resample_idxs)
        bootstrap_start_idxs_by_val_sample = resample(val_resample_idxs)

        return bootstrap_start_idxs_by_train_sample, X_by_sample[num_val:], U_by_sample[num_val:], O_by_sample[num_val:], output_by_sample[num_val:], \
               bootstrap_start_idxs_by_val_sample, X_by_sample[:num_val], U_by_sample[:num_val], O_by_sample[:num_val], output_by_sample[:num_val]

    def _save_tfrecords(self, tfrecords, bootstrap_start_idxs_by_sample,
                        X_by_sample, U_by_sample, O_by_sample, output_by_sample):
        if self.save_type == 'varlen':
            save_tfrecords = self._save_tfrecords_varlen
        elif self.save_type == 'fixedlen':
            save_tfrecords = self._save_tfrecords_fixedlen
        else:
            raise Exception('{0} not valid save type'.format(self.save_type))

        save_tfrecords(tfrecords, bootstrap_start_idxs_by_sample,
                       X_by_sample, U_by_sample, O_by_sample, output_by_sample)

    def _save_tfrecords_varlen(self, tfrecords, bootstrap_start_idxs_by_sample,
                               X_by_sample, U_by_sample, O_by_sample, output_by_sample):
        def _floatlist_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        def _int64list_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        writer = tf.python_io.TFRecordWriter(tfrecords)
        written = []
        for i, (bootstrap_start_idxs, X, U, O, output) in \
            enumerate(zip(bootstrap_start_idxs_by_sample, X_by_sample, U_by_sample, O_by_sample, output_by_sample)):

            num_start_idxs = sum([len(start_idxs) for start_idxs in bootstrap_start_idxs])
            if num_start_idxs == 0:
                continue
            written.append(i)

            feature = {
                'fname': _bytes_feature(os.path.splitext(os.path.basename(tfrecords))[0] + '_{0}'.format(i)),
                'X': _floatlist_feature(X.ravel().tolist()),
                'U': _floatlist_feature(U.ravel().tolist()),
                'O': _floatlist_feature(O.ravel().tolist()),
                'output': _int64list_feature(output.ravel().tolist()),
                'H': _int64list_feature([len(X)]),
            }
            for b, start_idxs in enumerate(bootstrap_start_idxs):
                feature['bootstrap_start_idxs_{0}'.format(b)] = _int64list_feature(start_idxs)
                feature['bootstrap_start_idxs_{0}_len'.format(b)] = _int64list_feature([len(start_idxs)])

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        writer.close()

    def _save_tfrecords_fixedlen(self, tfrecords, bootstrap_start_idxs_by_sample,
                        X_by_sample, U_by_sample, O_by_sample, output_by_sample):

        ### create all X / U / O / output for all bootstraps
        X_bootstrap = [[] for _ in xrange(self.num_bootstrap)]
        U_bootstrap = [[] for _ in xrange(self.num_bootstrap)]
        O_bootstrap = [[] for _ in xrange(self.num_bootstrap)]
        output_bootstrap = [[] for _ in xrange(self.num_bootstrap)]
        for bootstrap_start_idxs, X, U, O, output in \
            zip(bootstrap_start_idxs_by_sample, X_by_sample, U_by_sample, O_by_sample, output_by_sample):

            num_start_idxs = sum([len(start_idxs) for start_idxs in bootstrap_start_idxs])
            if num_start_idxs == 0:
                continue

            for b, start_idxs in enumerate(bootstrap_start_idxs):
                for start_idx in start_idxs:
                    idxs = slice(start_idx, start_idx + self.T)
                    X_bootstrap[b].append(X[idxs])
                    U_bootstrap[b].append(U[idxs])
                    O_bootstrap[b].append(O[idxs])
                    output_bootstrap[b].append(output[idxs])

        N = len(X_bootstrap[0])
        assert(np.all([len(l) == N for l in X_bootstrap + U_bootstrap + O_bootstrap + output_bootstrap]))

        def _floatlist_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        def _int64list_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        writer = tf.python_io.TFRecordWriter(tfrecords)
        for i in xrange(N):
            feature = {
                'fname': _bytes_feature(os.path.splitext(os.path.basename(tfrecords))[0] + '_{0}'.format(i)),
            }

            for b, (X, U, O, output) in enumerate(zip(X_bootstrap, U_bootstrap, O_bootstrap, output_bootstrap)):
                feature['X_{0}'.format(b)] = _floatlist_feature(np.ravel(X[i]).tolist())
                feature['U_{0}'.format(b)] = _floatlist_feature(np.ravel(U[i]).tolist())
                feature['O_{0}'.format(b)] = _floatlist_feature(np.ravel(O[i][0]).tolist())
                feature['output_{0}'.format(b)] = _int64list_feature([np.ravel(output[i])[-1]])

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        writer.close()

    def _save_preprocess(self, fname, X_by_sample, U_by_sample, O_by_sample):
        def calc_mean_cov(blank_by_sample):
            timesteps = 0
            mean_total = 0
            for blank in blank_by_sample:
                mean_total += blank.sum(axis=0)
                timesteps += len(blank)
            mean = mean_total / float(timesteps)

            cov_total = np.zeros((mean.shape[0], mean.shape[0]), dtype=float)
            for blank in blank_by_sample:
                for x in blank:
                    cov_total += np.outer(x - mean, x - mean)
            cov = cov_total / float(timesteps)
            
            return timesteps, mean, cov

        X_timesteps, X_mean, X_cov = calc_mean_cov(X_by_sample)
        U_timesteps, U_mean, U_cov = calc_mean_cov(U_by_sample)
        O_timesteps, O_mean, O_cov = calc_mean_cov(O_by_sample)

        assert(X_timesteps == U_timesteps and U_timesteps == O_timesteps)

        np.savez(fname, timesteps=X_timesteps,
                 X_mean=X_mean, X_cov=X_cov,
                 U_mean=U_mean, U_cov=U_cov,
                 O_mean=O_mean, O_cov=O_cov)

    def _compute_mean(self):
        """ Calculates mean and updates variable in graph """
        if len(self.preprocess_fnames) > 0:
            total_X_mean, total_U_mean, total_O_mean = 0, 0, 0
            total_X_cov, total_U_cov, total_O_cov = 0, 0, 0
            total_timesteps = 0
            for preprocess_fname in self.preprocess_fnames:
                d = np.load(preprocess_fname)
                total_timesteps += d['timesteps']
                total_X_mean += d['X_mean'] * d['timesteps']
                total_U_mean += d['U_mean'] * d['timesteps']
                total_O_mean += d['O_mean'] * d['timesteps']
                total_X_cov += d['X_cov'] * d['timesteps']
                total_U_cov += d['U_cov'] * d['timesteps']
                total_O_cov += d['O_cov'] * d['timesteps']

            X_mean = total_X_mean / float(total_timesteps)
            U_mean = total_U_mean / float(total_timesteps)
            O_mean = total_O_mean / float(total_timesteps)
            X_cov = total_X_cov / float(total_timesteps)
            U_cov = total_U_cov / float(total_timesteps)
            O_cov = total_O_cov / float(total_timesteps)
            if self.dX > 0:
                X_orth, X_eigs, _ = np.linalg.svd(X_cov)
                X_orth /= np.sqrt(X_eigs + 1e-5)
            else:
                X_orth = np.eye(self.dX, dtype=np.float32)
            if self.dU > 0:
                U_orth, U_eigs, _ = np.linalg.svd(U_cov)
                U_orth /= np.sqrt(U_eigs + 1e-5)
            else:
                U_orth = np.eye(self.dU, dtype=np.float32)
            if self.dO > 0 and self.use_O_orth:
                O_orth, O_eigs, _ = np.linalg.svd(O_cov)
                O_orth /= np.sqrt(O_eigs + 1e-5)
            else:
                O_orth = np.eye(self.dO, dtype=np.float32)
        else:
            X_mean = np.zeros(self.dX, dtype=np.float32)
            U_mean = np.zeros(self.dU, dtype=np.float32)
            O_mean = np.zeros(self.dO, dtype=np.float32)
            X_orth = np.eye(self.dX, dtype=np.float32)
            U_orth = np.eye(self.dU, dtype=np.float32)
            O_orth = np.eye(self.dO, dtype=np.float32)

        self.sess.run([self.d_mean['X_assign'], self.d_mean['U_assign'], self.d_mean['O_assign'],
                       self.d_orth['X_assign'], self.d_orth['U_assign'], self.d_orth['O_assign']],
                      feed_dict={self.d_mean['X_placeholder']: np.expand_dims(X_mean, 0),
                                 self.d_mean['U_placeholder']: np.expand_dims(U_mean, 0),
                                 self.d_mean['O_placeholder']: np.expand_dims(O_mean, 0),
                                 self.d_orth['X_placeholder']: X_orth,
                                 self.d_orth['U_placeholder']: U_orth,
                                 self.d_orth['O_placeholder']: O_orth})

    def add_data(self, npz_fnames):
        if self.aggregate_save_data:
            assert(self.reset_every_train)
            self.npz_fnames += sorted([f for f in npz_fnames if 'mean' not in f])
            tfrecords_train = self.npz_fnames[-1].replace('.npz', '_train_' + self._hash + '.tfrecords')
            tfrecords_val = self.npz_fnames[-1].replace('.npz', '_val_' + self._hash + '.tfrecords')
            preprocess_fname = self.npz_fnames[-1].replace('.npz', '_mean_' + self._hash + '.npz')

            self.logger.info('Current npz_fnames: {0}'.format(self.npz_fnames))
            self.logger.info('Generating and saving to {0}'.format(tfrecords_train))
            ### load samples
            random.shuffle(self.npz_fnames)
            start_idxs_by_sample, X_by_sample, U_by_sample, O_by_sample, output_by_sample = self._load_samples(self.npz_fnames)
            ### balance the data
            bootstrap_start_idxs_by_train_sample, X_by_train_sample, U_by_train_sample, O_by_train_sample, output_by_train_sample, \
            bootstrap_start_idxs_by_val_sample, X_by_val_sample, U_by_val_sample, O_by_val_sample, output_by_val_sample = \
                self._balance_data(start_idxs_by_sample, X_by_sample, U_by_sample, O_by_sample, output_by_sample)
            ### save train/val tfrecords
            self.logger.info('Saving tfrecords')
            self._save_tfrecords(tfrecords_train, bootstrap_start_idxs_by_train_sample,
                                 X_by_train_sample, U_by_train_sample, O_by_train_sample, output_by_train_sample)
            self._save_tfrecords(tfrecords_val, bootstrap_start_idxs_by_val_sample,
                                 X_by_val_sample, U_by_val_sample, O_by_val_sample, output_by_val_sample)
            ### save mean
            self._save_preprocess(preprocess_fname, X_by_train_sample, U_by_train_sample, O_by_train_sample)

            self.tfrecords_train_fnames = [tfrecords_train]
            self.tfrecords_val_fnames = [tfrecords_val]
            self.preprocess_fnames = [preprocess_fname]
        else:
            npz_fnames = [f for f in npz_fnames if 'mean' not in f]
            tfrecords_train_fnames = [f.replace('.npz', '_train_' + self._hash + '.tfrecords') for f in npz_fnames]
            tfrecords_val_fnames = [f.replace('.npz', '_val_' + self._hash + '.tfrecords') for f in npz_fnames]
            preprocess_fnames = [f.replace('.npz', '_mean_' + self._hash + '.npz') for f in npz_fnames]

            for npz, tfrecords_train, tfrecords_val, mean in zip(npz_fnames,
                                                                 tfrecords_train_fnames, tfrecords_val_fnames, preprocess_fnames):
                if not os.path.exists(tfrecords_train) or not os.path.exists(tfrecords_val):
                    self.logger.info('Generating and saving to {0}'.format(tfrecords_train))
                    ### load samples
                    start_idxs_by_sample, X_by_sample, U_by_sample, O_by_sample, output_by_sample = \
                        self._load_samples([npz])
                    ### balance the data
                    bootstrap_start_idxs_by_train_sample, X_by_train_sample, U_by_train_sample, O_by_train_sample, output_by_train_sample, \
                    bootstrap_start_idxs_by_val_sample, X_by_val_sample, U_by_val_sample, O_by_val_sample, output_by_val_sample = \
                        self._balance_data(start_idxs_by_sample, X_by_sample, U_by_sample, O_by_sample, output_by_sample)
                    ### save train/val tfrecords
                    self._save_tfrecords(tfrecords_train, bootstrap_start_idxs_by_train_sample,
                                         X_by_train_sample, U_by_train_sample, O_by_train_sample, output_by_train_sample)
                    self._save_tfrecords(tfrecords_val, bootstrap_start_idxs_by_val_sample,
                                         X_by_val_sample, U_by_val_sample, O_by_val_sample, output_by_val_sample)
                    ### save mean
                    self._save_preprocess(mean, X_by_train_sample, U_by_train_sample, O_by_train_sample)

                else:
                    self.logger.info('{0} already exists'.format(tfrecords_train))

            self.npz_fnames += npz_fnames
            self.tfrecords_train_fnames += tfrecords_train_fnames
            self.tfrecords_val_fnames += tfrecords_val_fnames
            self.preprocess_fnames += preprocess_fnames

    #############
    ### Graph ###
    #############

    def _graph_inputs_outputs_from_file(self, name):
        if self.save_type == 'varlen':
            graph_inputs_outputs_from_file = self._graph_inputs_outputs_from_file_varlen
        elif self.save_type == 'fixedlen':
            graph_inputs_outputs_from_file = self._graph_inputs_outputs_from_file_fixedlen
        else:
            raise Exception('{0} is not valid save type'.format(self.save_type))

        return graph_inputs_outputs_from_file(name)

    def _graph_inputs_outputs_from_file_varlen(self, name):
        with tf.name_scope(name + '_file_input'):
            filename_place = tf.placeholder(tf.string)
            filename_var = tf.get_variable(name + '_fnames', initializer=filename_place, validate_shape=False, trainable=False)

            ### create file queue
            filename_queue = tf.train.string_input_producer(filename_var,
                                                            num_epochs=None,
                                                            shuffle=True)

            ### read and decode
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)

            features = {
                'fname': tf.FixedLenFeature([], tf.string),
                'X': tf.VarLenFeature(tf.float32),
                'U': tf.VarLenFeature(tf.float32),
                'O': tf.VarLenFeature(tf.float32),
                'output': tf.VarLenFeature(tf.int64),
                'H': tf.FixedLenFeature([], tf.int64)
            }
            for b in xrange(self.num_bootstrap):
                features['bootstrap_start_idxs_{0}'.format(b)] = tf.VarLenFeature(tf.int64)
                features['bootstrap_start_idxs_{0}_len'.format(b)] = tf.FixedLenFeature([], tf.int64)

            batch = tf.train.batch([serialized_example], self.batch_size, capacity=self.batch_size)
            parsed_examples = tf.parse_example(batch, features)

            ### densify and reshape
            fname_batch = parsed_examples['fname']
            H_batch = tf.cast(parsed_examples['H'], tf.int32)
            H_max = tf.reduce_max(H_batch)
            X_batch = tf.sparse_tensor_to_dense(parsed_examples['X'])
            U_batch = tf.sparse_tensor_to_dense(parsed_examples['U'])
            O_batch = tf.sparse_tensor_to_dense(parsed_examples['O'])
            output_batch = tf.sparse_tensor_to_dense(parsed_examples['output'])

            self.tf_debug['H_batch'] = H_batch
            self.tf_debug['H_max'] = H_max
            self.tf_debug['X_batch'] = X_batch
            self.tf_debug['U_batch'] = U_batch
            self.tf_debug['O_batch'] = O_batch
            self.tf_debug['output_batch'] = output_batch

            for tensor in (X_batch, U_batch, O_batch, output_batch):
                tensor.set_shape([self.batch_size, None])

            # [# bootstrap, batch size]
            bootstrap_start_idxs_len_batch = []
            bootstrap_start_idxs_batch = []
            for b in xrange(self.num_bootstrap):
                start_idxs_len_batch = tf.cast(parsed_examples['bootstrap_start_idxs_{0}_len'.format(b)], tf.int32)
                start_idxs_batch = tf.sparse_tensor_to_dense(parsed_examples['bootstrap_start_idxs_{0}'.format(b)])
                start_idxs_batch.set_shape([self.batch_size, None])
                # start_idxs_batch = [tf.slice(si, [0], [si_len]) for si, si_len in
                #                     zip(tf.unpack(start_idxs_batch), tf.unpack(start_idxs_len_batch))]

                bootstrap_start_idxs_len_batch.append(tf.unpack(start_idxs_len_batch))
                bootstrap_start_idxs_batch.append(tf.unpack(start_idxs_batch))
            # [batch size, # bootstrap]
            bootstrap_start_idxs_len_batch = np.array(bootstrap_start_idxs_len_batch).T.tolist()
            bootstrap_start_idxs_batch = np.array(bootstrap_start_idxs_batch).T.tolist()

            ### iterate one example at a time
            bootstrap_X_inputs = [[] for _ in xrange(self.num_bootstrap)]
            bootstrap_U_inputs = [[] for _ in xrange(self.num_bootstrap)]
            bootstrap_O_inputs = [[] for _ in xrange(self.num_bootstrap)]
            bootstrap_outputs = [[] for _ in xrange(self.num_bootstrap)]
            for i, (H, X, U, O, output, bootstrap_start_idxs_len, bootstrap_start_idxs) in \
                    enumerate(zip(tf.unpack(H_batch, name='unpack_H'), tf.unpack(X_batch, name='unpack_X'), tf.unpack(U_batch, name='unpack_U'), tf.unpack(O_batch, name='unpack_O'), tf.unpack(output_batch, name='unpack_output'),
                        bootstrap_start_idxs_len_batch, bootstrap_start_idxs_batch)):

                # tf.Print(H, [H], 'H = ', summarize=100)

                X = tf.reshape(X, tf.pack([H_max, self.dX]), name='reshape_X_{0}'.format(i))
                U = tf.reshape(U, tf.pack([H_max, self.dU]), name='reshape_U_{0}'.format(i))
                O = tf.reshape(O, tf.pack([H_max, self.dO]), name='reshape_O_{0}'.format(i))
                output = tf.reshape(output, tf.pack([H_max]), name='reshape_output_{0}'.format(i))

                self.tf_debug['X_{0}'.format(i)] = X
                self.tf_debug['U_{0}'.format(i)] = U
                self.tf_debug['O_{0}'.format(i)] = O
                self.tf_debug['output_{0}'.format(i)] = output

                ### iterate through each bootstrap
                for b, (start_idxs_len, start_idxs) in enumerate(zip(bootstrap_start_idxs_len, bootstrap_start_idxs)):
                    start_idxs = tf.slice(start_idxs, [0], [start_idxs_len])
                    self.tf_debug['start_idxs_before_{0}_b{1}'.format(i, b)] = start_idxs
                    start_idxs = tf.expand_dims(start_idxs, 1)
                    start_idxs = tf.tile(start_idxs, (1, self.T))
                    start_idxs += tf.constant(range(self.T), dtype=tf.int64)

                    self.tf_debug['start_idxs_len_{0}_b{1}'.format(i, b)] = start_idxs_len
                    self.tf_debug['start_idxs_{0}_b{1}'.format(i, b)] = start_idxs

                    bootstrap_X_inputs[b].append(tf.gather(X, start_idxs))
                    bootstrap_U_inputs[b].append(tf.gather(U, start_idxs))
                    bootstrap_O_inputs[b].append(tf.gather(O, start_idxs[:, 0])) # only first observation
                    bootstrap_outputs[b].append(tf.expand_dims(tf.gather(output, start_idxs[:, self.T-1]), 1)) # only last output

                    self.tf_debug['X_{0}_b{1}'.format(i, b)] = bootstrap_X_inputs[b][-1]
                    self.tf_debug['U_{0}_b{1}'.format(i, b)] = bootstrap_U_inputs[b][-1]
                    self.tf_debug['O_{0}_b{1}'.format(i, b)] = bootstrap_O_inputs[b][-1]
                    self.tf_debug['output_{0}_b{1}'.format(i, b)] = bootstrap_outputs[b][-1]

            for b in xrange(self.num_bootstrap):
                bootstrap_X_inputs[b] = tf.concat(0, bootstrap_X_inputs[b])
                bootstrap_U_inputs[b] = tf.concat(0, bootstrap_U_inputs[b])
                bootstrap_O_inputs[b] = tf.concat(0, bootstrap_O_inputs[b])
                bootstrap_outputs[b] = tf.concat(0, bootstrap_outputs[b])

        return fname_batch, bootstrap_X_inputs, bootstrap_U_inputs, bootstrap_O_inputs, bootstrap_outputs, \
               filename_queue, filename_place, filename_var

    def _graph_inputs_outputs_from_file_fixedlen(self, name):
        with tf.name_scope(name + '_file_input'):
            filename_place = tf.placeholder(tf.string)
            filename_var = tf.get_variable(name + '_fnames', initializer=filename_place, validate_shape=False, trainable=False)

            ### create file queue
            filename_queue = tf.train.string_input_producer(filename_var,
                                                            num_epochs=None,
                                                            shuffle=True)

            ### read and decode
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)

            features = {
                'fname': tf.FixedLenFeature([], tf.string)
            }

            for b in xrange(self.num_bootstrap):
                features['X_{0}'.format(b)] = tf.FixedLenFeature([self.dX * self.T], tf.float32)
                features['U_{0}'.format(b)] = tf.FixedLenFeature([self.dU * self.T], tf.float32)
                features['O_{0}'.format(b)] = tf.FixedLenFeature([self.dO], tf.float32)
                features['output_{0}'.format(b)] = tf.FixedLenFeature([1], tf.int64)

            parsed_example = tf.parse_single_example(serialized_example, features=features)

            fname = parsed_example['fname']
            bootstrap_X_input = [tf.reshape(parsed_example['X_{0}'.format(b)], (self.T, self.dX))
                                 for b in xrange(self.num_bootstrap)]
            bootstrap_U_input = [tf.reshape(parsed_example['U_{0}'.format(b)], (self.T, self.dU))
                                 for b in xrange(self.num_bootstrap)]
            bootstrap_O_input = [parsed_example['O_{0}'.format(b)] for b in xrange(self.num_bootstrap)]
            bootstrap_output = [parsed_example['output_{0}'.format(b)] for b in xrange(self.num_bootstrap)]

            shuffled = tf.train.shuffle_batch(
                [fname] + bootstrap_X_input + bootstrap_U_input + bootstrap_O_input + bootstrap_output,
                batch_size=self.batch_size,
                capacity=10*self.batch_size + 3 * self.batch_size,
                min_after_dequeue=10*self.batch_size)

            fname_batch = shuffled[0]
            bootstrap_X_inputs = shuffled[1:1+self.num_bootstrap]
            bootstrap_U_inputs = shuffled[1+self.num_bootstrap:1+2*self.num_bootstrap]
            bootstrap_O_inputs = shuffled[1+2*self.num_bootstrap:1+3*self.num_bootstrap]
            bootstrap_outputs = shuffled[1+3*self.num_bootstrap:1+4*self.num_bootstrap]

        return fname_batch, bootstrap_X_inputs, bootstrap_U_inputs, bootstrap_O_inputs, bootstrap_outputs,\
               filename_queue, filename_place, filename_var

    def _graph_inputs_outputs_from_placeholders(self):
        with tf.variable_scope('feed_input'):
            bootstrap_X_inputs = [tf.placeholder('float', [None, self.T, self.dX]) for _ in xrange(self.num_bootstrap)]
            bootstrap_U_inputs = [tf.placeholder('float', [None, self.T, self.dU]) for _ in xrange(self.num_bootstrap)]
            bootstrap_O_inputs = [tf.placeholder('float', [None, self.dO]) for _ in xrange(self.num_bootstrap)]
            bootstrap_outputs = [tf.placeholder('float', [None]) for _ in xrange(self.num_bootstrap)]

        return bootstrap_X_inputs, bootstrap_U_inputs, bootstrap_O_inputs, bootstrap_outputs

    def _graph_cost(self, name, bootstrap_output_mats, bootstrap_outputs, reg=0.):
        with tf.name_scope(name + '_cost_and_err'):
            costs = []
            num_coll = 0
            num_errs_on_coll = 0
            num_nocoll = 0
            num_errs_on_nocoll = 0

            for b, (output_mat_b, output_b) in enumerate(zip(bootstrap_output_mats, bootstrap_outputs)):
                output_b = tf.to_float(output_b)
                output_pred_b = tf.nn.sigmoid(output_mat_b)

                ### cost
                with tf.name_scope('cost_b{0}'.format(b)):
                    cross_entropy_b = tf.nn.sigmoid_cross_entropy_with_logits(output_mat_b, output_b)
                    costs.append(cross_entropy_b)
                ### accuracy
                with tf.name_scope('err_b{0}'.format(b)):
                    output_geq_b = tf.cast(tf.greater_equal(output_pred_b, 0.5), tf.float32)
                    output_incorrect_b = tf.cast(tf.not_equal(output_geq_b, output_b), tf.float32)

                    ### coll err
                    num_coll += tf.reduce_sum(output_b)
                    num_errs_on_coll += tf.reduce_sum(output_b * output_incorrect_b)

                    ### nocoll err
                    num_nocoll += tf.reduce_sum(1 - output_b)
                    num_errs_on_nocoll += tf.reduce_sum((1 - output_b) * output_incorrect_b)

            with tf.name_scope('total'):
                cross_entropy = tf.reduce_mean(tf.concat(0, costs))
                weight_decay = reg * tf.add_n(tf.get_collection('weight_decays'))
                cost = cross_entropy + weight_decay
                err = (1. / tf.cast(num_coll + num_nocoll, tf.float32)) * (num_errs_on_coll + num_errs_on_nocoll)
                err_on_coll = tf.cond(num_coll > 0,
                                      lambda: (1. / tf.cast(num_coll, tf.float32)) * num_errs_on_coll,
                                      lambda: tf.constant(np.nan))
                err_on_nocoll = tf.cond(num_nocoll > 0,
                                        lambda: (1. / tf.cast(num_nocoll, tf.float32)) * num_errs_on_nocoll,
                                        lambda: tf.constant(np.nan))


        return cost, cross_entropy, err, err_on_coll, err_on_nocoll

    def _graph_optimize(self, cost):
        vars_before = tf.global_variables()

        with tf.name_scope('optimizer'):
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            optimizer = opt.minimize(cost)
            grad = opt.compute_gradients(cost)

        vars_after = tf.global_variables()
        optimizer_vars = list(set(vars_after).difference(set(vars_before)))

        return optimizer, grad, optimizer_vars

    def _graph_init_vars(self):
        self.sess.run(tf.global_variables_initializer(),
                      feed_dict=dict([(p, []) for p in (self.d_train['queue_placeholder'],
                                                        self.d_val['queue_placeholder'])]))
        self._compute_mean()

    def _graph_setup(self):
        """ Only call once """

        tf.reset_default_graph()

        self.d_mean = dict()
        self.d_orth = dict()
        self.d_train = dict()
        self.d_val = dict()
        self.d_eval = dict()

        ### data stuff
        with tf.variable_scope('means', reuse=False):
            # X
            self.d_mean['X_placeholder'] = tf.placeholder(tf.float32, shape=(1, self.dX))
            self.d_mean['X_var'] = tf.get_variable('X_mean', shape=[1, self.dX], trainable=False, dtype=tf.float32,
                                                   initializer=tf.constant_initializer(np.zeros((1, self.dX))))
            self.d_mean['X_assign'] = tf.assign(self.d_mean['X_var'], self.d_mean['X_placeholder'])
            # U
            self.d_mean['U_placeholder'] = tf.placeholder(tf.float32, shape=(1, self.dU))
            self.d_mean['U_var'] = tf.get_variable('U_mean', shape=[1, self.dU], trainable=False, dtype=tf.float32,
                                                   initializer=tf.constant_initializer(np.zeros((1, self.dU))))
            self.d_mean['U_assign'] = tf.assign(self.d_mean['U_var'], self.d_mean['U_placeholder'])
            # O
            self.d_mean['O_placeholder'] = tf.placeholder(tf.float32, shape=(1, self.dO))
            self.d_mean['O_var'] = tf.get_variable('O_mean', shape=[1, self.dO], trainable=False, dtype=tf.float32,
                                                   initializer=tf.constant_initializer(np.zeros((1, self.dO))))
            self.d_mean['O_assign'] = tf.assign(self.d_mean['O_var'], self.d_mean['O_placeholder'])
        with tf.variable_scope('orths', reuse=False):
            # X
            self.d_orth['X_placeholder'] = tf.placeholder(tf.float32, shape=(self.dX, self.dX))
            self.d_orth['X_var'] = tf.get_variable('X_orth', shape=[self.dX, self.dX], trainable=False, dtype=tf.float32,
                                                   initializer=tf.constant_initializer(np.zeros((self.dX, self.dX))))
            self.d_orth['X_assign'] = tf.assign(self.d_orth['X_var'], self.d_orth['X_placeholder'])
            # U
            self.d_orth['U_placeholder'] = tf.placeholder(tf.float32, shape=(self.dU, self.dU))
            self.d_orth['U_var'] = tf.get_variable('U_orth', shape=[self.dU, self.dU], trainable=False, dtype=tf.float32,
                                                   initializer=tf.constant_initializer(np.zeros((self.dU, self.dU))))
            self.d_orth['U_assign'] = tf.assign(self.d_orth['U_var'], self.d_orth['U_placeholder'])
            # O
            self.d_orth['O_placeholder'] = tf.placeholder(tf.float32, shape=(self.dO, self.dO))
            self.d_orth['O_var'] = tf.get_variable('O_orth', shape=[self.dO, self.dO], trainable=False, dtype=tf.float32,
                                                   initializer=tf.constant_initializer(np.zeros((self.dO, self.dO))))
            self.d_orth['O_assign'] = tf.assign(self.d_orth['O_var'], self.d_orth['O_placeholder'])

        ### prepare for training
        for i, (name, d) in enumerate((('train', self.d_train), ('val', self.d_val))):
            d['fnames'], d['X_inputs'], d['U_inputs'], d['O_inputs'], d['outputs'], \
            d['queue'], d['queue_placeholder'], d['queue_var'] = self._graph_inputs_outputs_from_file(name)
            _, _, _, _, d['output_mats'], _ = self._graph_inference(name, self.T,
                                                                    d['X_inputs'], d['U_inputs'], d['O_inputs'],
                                                                    self.d_mean['X_var'], self.d_orth['X_var'],
                                                                    self.d_mean['U_var'], self.d_orth['U_var'],
                                                                    self.d_mean['O_var'], self.d_orth['O_var'],
                                                                    self.dropout, params,
                                                                    reuse=i>0, random_seed=self.random_seed,
                                                                    tf_debug=self.tf_debug)
            d['cost'], d['cross_entropy'], d['err'], d['err_coll'], d['err_nocoll'] = \
                self._graph_cost(name, d['output_mats'], d['outputs'], reg=self.reg)

        ### optimizer
        self.d_train['optimizer'], self.d_train['grads'], self.d_train['optimizer_vars'] = \
            self._graph_optimize(self.d_train['cost'])

        ### prepare for eval
        self.d_eval['X_inputs'], self.d_eval['U_inputs'], self.d_eval['O_inputs'], self.d_eval['outputs'] = \
            self._graph_inputs_outputs_from_placeholders()
        self.d_eval['output_pred_mean'], self.d_eval['output_pred_std'], self.d_eval['output_mat_mean'], \
        self.d_eval['output_mat_std'], _, self.d_eval['dropout_placeholders'] = \
            self._graph_inference('eval', self.T,
                                  self.d_eval['X_inputs'], self.d_eval['U_inputs'], self.d_eval['O_inputs'],
                                  self.d_mean['X_var'], self.d_orth['X_var'],
                                  self.d_mean['U_var'], self.d_orth['U_var'],
                                  self.d_mean['O_var'], self.d_orth['O_var'],
                                  self.dropout, params,
                                  reuse=True, random_seed=self.random_seed)

        ### initialize
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_fraction)
        config = tf.ConfigProto(gpu_options=gpu_options,
                                log_device_placement=False,
                                allow_soft_placement=True)
        # config.intra_op_parallelism_threads = 1
        # config.inter_op_parallelism_threads = 1
        self.sess = tf.Session(config=config)
        self._graph_init_vars()

        # Set logs writer into folder /tmp/tensorflow_logs
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter('/tmp', graph_def=self.sess.graph_def)

        self.saver = tf.train.Saver(max_to_keep=None)

    @abc.abstractmethod
    def _get_old_graph_inference(self):
        raise NotImplementedError('Implement in subclass')

    # @staticmethod
    # @abc.abstractmethod

    # def _graph_inference(name, T, bootstrap_X_inputs, bootstrap_U_inputs, bootstrap_O_inputs,
    #                      X_mean, U_mean, O_mean, dropout, meta_data,
    #                      reuse=False, random_seed=None, finalize=True, tf_debug={}):
    #     raise NotImplementedError('Implement in subclass')

    ################
    ### Training ###
    ################

    def _create_input(self, X, U, O):
        return X, U, O

    def _create_output(self, output):
        return output

    def _get_epoch(self, fnames_dict, fnames_value):
        for fname in fnames_value:
            fnames_dict[fname] += 1

        epoch = max(fnames_dict.values()) - 1
        return epoch

    def _flush_queue(self):
        for tfrecords_fnames, queue in ((self.tfrecords_train_fnames, self.d_train['queue']),
                                        (self.tfrecords_val_fnames, self.d_val['queue'])):
            while not np.all([(fname in tfrecords_fnames) for fname in
                              self.sess.run(queue.dequeue_many(10*self.batch_size))]):
                pass

    def train(self, prev_model_file=None, new_model_file=None, **kwargs):
        epochs = kwargs.get('epochs', self.epochs)

        if prev_model_file is not None and not self.reset_every_train:
            self.load(prev_model_file)
        else:
            self._graph_init_vars()
        self._compute_mean()

        self.sess.run([tf.assign(self.d_train['queue_var'], self.tfrecords_train_fnames, validate_shape=False),
                       tf.assign(self.d_val['queue_var'], self.tfrecords_val_fnames, validate_shape=False)])

        if not hasattr(self, 'coord'):
            assert(not hasattr(self, 'threads'))
            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        self.logger.debug('Flushing queue')
        self._flush_queue()

        ### create plotter
        plotter = MLPlotter(self.save_dir,
                            {
                                'err': {
                                    'title': 'Error',
                                    'subplot': 0,
                                    'color': 'k',
                                    'ylabel': 'Percentage'
                                },
                                'err_coll': {
                                    'title': 'Error Collision',
                                    'subplot': 1,
                                    'color': 'r',
                                    'ylabel': 'Percentage'
                                },
                                'err_nocoll': {
                                    'title': 'Error no collision',
                                    'subplot': 2,
                                    'color': 'g',
                                    'ylabel': 'Percentage'
                                },
                                'cost': {
                                    'title': 'Cost',
                                    'subplot': 3,
                                    'color': 'k',
                                    'ylabel': 'cost'
                                },
                                'cross_entropy': {
                                    'subplot': 3,
                                    'color': 'm'
                                }
                            })

        ### train
        train_values = defaultdict(list)
        train_nums = defaultdict(float)

        train_fnames_dict = defaultdict(int)
        val_fnames_dict = defaultdict(int)

        train_epoch = -1
        new_train_epoch = 0
        val_epoch = 0
        step = 0
        save_start = time.time()
        epoch_start = time.time()
        while train_epoch < epochs and step < self.steps:
            if step == 0:
                for _ in xrange(10): print('')

            ### validation
            if new_train_epoch != train_epoch:
                val_values = defaultdict(list)
                val_nums = defaultdict(float)

                self.logger.debug('\tComputing validation...')
                while True:
                    val_cost, val_cross_entropy, \
                    val_err, val_err_coll, val_err_nocoll, \
                    val_fnames, val_outputs = \
                        self.sess.run([self.d_val['cost'], self.d_val['cross_entropy'],
                                       self.d_val['err'], self.d_val['err_coll'], self.d_val['err_nocoll'],
                                       self.d_val['fnames'], self.d_val['outputs']])

                    val_values['cost'].append(val_cost)
                    val_values['cross_entropy'].append(val_cross_entropy)
                    val_values['err'].append(val_err)
                    if not np.isnan(val_err_coll): val_values['err_coll'].append(val_err_coll)
                    if not np.isnan(val_err_nocoll): val_values['err_nocoll'].append(val_err_nocoll)
                    val_nums['coll'] += np.sum(np.concatenate(val_outputs))
                    val_nums['nocoll'] += np.sum(1 - np.concatenate(val_outputs))

                    new_val_epoch = self._get_epoch(val_fnames_dict, val_fnames)
                    if new_val_epoch != val_epoch:
                        val_epoch = new_val_epoch
                        break

                plotter.add_val('err', np.mean(val_values['err']))
                plotter.add_val('err_coll', np.mean(val_values['err_coll']))
                plotter.add_val('err_nocoll', np.mean(val_values['err_nocoll']))
                plotter.add_val('cost', np.mean(val_values['cost']))
                plotter.add_val('cross_entropy', np.mean(val_values['cross_entropy']))
                plotter.plot()

                self.logger.debug(
                    'Epoch {0:d},  error: {1:5.2f}%,  error coll: {2:5.2f}%,  error nocoll: {3:5.2f}%,  pct coll: {4:4.1f}%,  cost: {5:4.2f}, ce: {6:4.2f} ({7:.2f} s per {8:04d} samples)'.format(
                        train_epoch + 1,
                        100 * np.mean(val_values['err']),
                        100 * np.mean(val_values['err_coll']),
                        100 * np.mean(val_values['err_nocoll']),
                        100 * val_nums['coll'] / (val_nums['coll'] + val_nums['nocoll']),
                        np.mean(val_values['cost']),
                        np.mean(val_values['cross_entropy']),
                        time.time() - epoch_start,
                        step * self.batch_size / (train_epoch + 1) if train_epoch >= 0 else 0))
                fnames_condensed = defaultdict(int)
                for k, v in train_fnames_dict.items():
                    fnames_condensed[k.split(self._hash)[0]] += v
                for k, v in sorted(fnames_condensed.items(), key=lambda x: x[1]):
                    self.logger.debug('\t\t\t{0} : {1}'.format(k, v))

                epoch_start = time.time()

                ### save model
                self.save(new_model_file)

            train_epoch = new_train_epoch

            ### train
            _, train_cost, train_cross_entropy, \
            train_err, train_err_coll, train_err_nocoll, \
            train_fnames, train_outputs = self.sess.run([self.d_train['optimizer'],
                                                         self.d_train['cost'],
                                                         self.d_train['cross_entropy'],
                                                         self.d_train['err'],
                                                         self.d_train['err_coll'],
                                                         self.d_train['err_nocoll'],
                                                         self.d_train['fnames'],
                                                         self.d_train['outputs']])

            train_values['cost'].append(train_cost)
            train_values['cross_entropy'].append(train_cross_entropy)
            train_values['err'].append(train_err)
            if not np.isnan(train_err_coll): train_values['err_coll'].append(train_err_coll)
            if not np.isnan(train_err_nocoll): train_values['err_nocoll'].append(train_err_nocoll)
            train_nums['coll'] += np.sum(np.concatenate(train_outputs))
            train_nums['nocoll'] += np.sum(1 - np.concatenate(train_outputs))

            new_train_epoch = self._get_epoch(train_fnames_dict, train_fnames)

            # Print an overview fairly often.
            if step % self.display_batch == 0 and step > 0:
                plotter.add_train('err', step * self.batch_size, np.mean(train_values['err']))
                if len(train_values['err_coll']) > 0:
                    plotter.add_train('err_coll', step * self.batch_size, np.mean(train_values['err_coll']))
                if len(train_values['err_nocoll']) > 0:
                    plotter.add_train('err_nocoll', step * self.batch_size, np.mean(train_values['err_nocoll']))
                plotter.add_train('cost', step * self.batch_size, np.mean(train_values['cost']))
                plotter.add_train('cross_entropy', step * self.batch_size, np.mean(train_values['cross_entropy']))
                plotter.plot()

                self.logger.debug('\tepoch {0:d}/{1:d}, step pct: {2:.1f}%,  error: {3:5.2f}%,  error coll: {4:5.2f}%,  error nocoll: {5:5.2f}%,  pct coll: {6:4.1f}%,  cost: {7:4.2f}, ce: {8:4.2f}'.format(
                    train_epoch, self.epochs,
                    100 * step / float(self.steps),
                    100 * np.mean(train_values['err']),
                    100 * np.mean(train_values['err_coll']),
                    100 * np.mean(train_values['err_nocoll']),
                    100 * train_nums['coll'] / (train_nums['coll'] + train_nums['nocoll']),
                    np.mean(train_values['cost']),
                    np.mean(train_values['cross_entropy'])))

                train_values = defaultdict(list)
                train_nums = defaultdict(float)

            if time.time() - save_start > 60.:
                plotter.save(os.path.dirname(new_model_file))
                save_start = time.time()

            step += 1

        self.save(new_model_file)
        plotter.save(os.path.dirname(new_model_file))
        plotter.close()

    ##################
    ### Evaluating ###
    ##################

    def eval(self, X, U, O, num_avg=1, pre_activation=False):
        return self.eval_batch([X], [U], [O], num_avg=num_avg, pre_activation=pre_activation)

    def eval_batch(self, Xs, Us, Os, num_avg=1, pre_activation=False):
        X_inputs, U_inputs, O_inputs = [], [], []
        for X, U, O in zip(Xs, Us, Os):
            assert(len(X) >= self.T)
            assert(len(U) >= self.T)
            assert(len(O) >= 1)

            X_input, U_input, O_input = self._create_input(X, U, O)
            X_input = X_input[:self.T]
            U_input = U_input[:self.T]
            O_input = O_input[0]
            assert(not np.isnan(X_input).any())
            assert(not np.isnan(U_input).any())
            assert(not np.isnan(O_input).any())
            for _ in xrange(num_avg):
                X_inputs.append(X_input)
                U_inputs.append(U_input)
                O_inputs.append(O_input)

        feed = {}
        for b in xrange(self.num_bootstrap):
            feed[self.d_eval['X_inputs'][b]] = X_inputs
            feed[self.d_eval['U_inputs'][b]] = U_inputs
            feed[self.d_eval['O_inputs'][b]] = O_inputs
        # want dropout for each X/U/O to be the same
        # want dropout for each num_avg to be different
        # -->
        # create num_avg different dropout for each dropout mask
        # use same dropout mask for each X/U/O
        #
        # if num_avg = 3
        # 0 1 2 0 1 2 0 1 2
        for dropout_placeholder in self.d_eval['dropout_placeholders']:
            length = dropout_placeholder.get_shape()[1].value
            feed[dropout_placeholder] = [(1/self.dropout) * (np.random.random(length) < self.dropout).astype(float)
                                         for _ in xrange(num_avg)] * len(Xs)

        if pre_activation:
            output_pred_mean, output_pred_std = self.sess.run([self.d_eval['output_mat_mean'],
                                                               self.d_eval['output_mat_std']],
                                                              feed_dict=feed)
        else:
            output_pred_mean, output_pred_std = self.sess.run([self.d_eval['output_pred_mean'],
                                                               self.d_eval['output_pred_std']],
                                                              feed_dict=feed)

        if num_avg > 1:
            mean, std = [], []
            for i in xrange(len(Xs)):
                mean.append(output_pred_mean[i*num_avg:(i+1)*num_avg].mean(axis=0))
                std.append(output_pred_std[i*num_avg:(i+1)*num_avg].mean(axis=0))
            output_pred_mean = np.array(mean)
            output_pred_std = np.array(std)

            assert(len(output_pred_mean) == len(Xs))
            assert(len(output_pred_std) == len(Xs))

        assert((output_pred_std >= 0).all())

        return output_pred_mean, output_pred_std

    def eval_sample(self, sample):
        X = sample.get_X()[:self.T, self.X_idxs(sample._meta_data)]
        U = sample.get_U()[:self.T, self.U_idxs(sample._meta_data)]
        O = sample.get_O()[:self.T, self.O_idxs(sample._meta_data)]

        return self.eval(X, U, O)

    def eval_sample_batch(self, samples, num_avg=1, pre_activation=False):
        Xs = [sample.get_X()[:self.T, self.X_idxs(sample._meta_data)] for sample in samples]
        Us = [sample.get_U()[:self.T, self.U_idxs(sample._meta_data)] for sample in samples]
        Os = [sample.get_O()[:self.T, self.O_idxs(sample._meta_data)] for sample in samples]

        return self.eval_batch(Xs, Us, Os, num_avg=num_avg, pre_activation=pre_activation)

    def eval_jac(self, X, U, O):
        assert(len(X) >= self.T)
        assert(len(U) >= self.T)
        assert(len(O) >= self.T)

        x_input, u_input, o_input = self._create_input(X, U, O)
        feed = {}
        for b in xrange(self.num_bootstrap):
            feed.update({
                self.x_inputs[b]: [x_input],
                self.u_inputs[b]: [u_input],
                self.o_inputs[b]: [o_input]
            })

        all_grads = self.sess.run(self.x_grads + self.u_grads, feed_dict=feed)
        jac_x, jac_u = None, None
        if self.dX > 0:
            jac_x = np.vstack([all_grads.pop(0) for _ in xrange(self.T)])
        if self.dU > 0:
            jac_u = np.vstack([all_grads.pop(0) for _ in xrange(self.T)])

        return jac_x, jac_u

    def eval_jac_sample(self, sample):
        X = sample.get_X()[:self.T, self.X_idxs(sample._meta_data)]
        U = sample.get_U()[:self.T, self.U_idxs(sample._meta_data)]
        O = sample.get_O()[:self.T, self.O_idxs(sample._meta_data)]

        return self.eval_jac(X, U, O)

    def eval_jac_input(self, x_input, u_input, o_input):
        feed = {}
        for b in xrange(self.num_bootstrap):
            feed.update({
                self.x_inputs[b]: [x_input],
                self.u_inputs[b]: [u_input],
                self.o_inputs[b]: [o_input]
            })
        all_grads = self.sess.run(self.x_grads + self.u_grads, feed_dict=feed)
        jac_x, jac_u = None, None
        if self.dX > 0:
            jac_x = np.vstack([all_grads.pop(0) for _ in xrange(self.T)])
        if self.dU > 0:
            jac_u = np.vstack([all_grads.pop(0) for _ in xrange(self.T-1)])

        return jac_x, jac_u

    #############################
    ### Load/save/reset/close ###
    #############################

    def load(self, model_file):
        self.saver.restore(self.sess, model_file)

    def save(self, model_file):
        self.saver.save(self.sess, model_file, write_meta_graph=False)

    def close(self):
        """ Release tf session """
        if hasattr(self, 'coord'):
            assert(hasattr(self, 'threads'))
            self.coord.request_stop()
            self.coord.join(self.threads)
        self.sess.close()
        self.sess = None

    @staticmethod
    def checkpoint_exists(model_file):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_file))
        return ckpt is not None and model_file in ckpt.model_checkpoint_path

