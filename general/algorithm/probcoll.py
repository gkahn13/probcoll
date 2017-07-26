import abc
import os, pickle, time, shutil
import numpy as np
import multiprocessing
from general.algorithm.probcoll_model import ProbcollModel
from config import params

from general.utility.logger import get_logger
from general.state_info.sample import Sample
class Probcoll:
    __metaclass__ = abc.ABCMeta

    def __init__(self, save_dir=None, data_dir=None):
        self._planner_type = params['planning']['planner_type']
        self._async_on = False
        self._asynchronous = params['probcoll']['asynchronous_training']
        self._jobs = []
        if save_dir is None:
            self._save_dir = os.path.join(params['exp_dir'], params['exp_name'])
        else:
            self._save_dir = save_dir
        self._data_dir = data_dir
        self._setup()
        self._logger = get_logger(
            self.__class__.__name__,
            params['probcoll']['logger'],
            os.path.join(self._save_dir, 'dagger.txt'))
        self._mpc_policy = self._create_mpc()
        shutil.copy(params['yaml_path'], os.path.join(self._save_dir, '{0}.yaml'.format(params['exp_name'])))

    @abc.abstractmethod
    def _setup(self):
        ### load prediction neural net
        self.probcoll_model = None
        self._asynchprobcoll_model = None
        self._max_iter = None
        self._agent = None
        raise NotImplementedError('Implement in subclass')

    #########################
    ### Create controller ###
    #########################

    @abc.abstractmethod
    def _create_mpc(self):
        raise NotImplementedError('Implement in subclass')

    ####################
    ### Save methods ###
    ####################

    def _itr_dir(self, itr, create=True):
        assert(type(itr) is int)
        dir = os.path.join(self._save_dir, 'itr{0}'.format(itr))
        if not create:
            return dir
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir

    def _itr_samples_file(self, itr, prefix='', create=True):
        return os.path.join(self._itr_dir(itr, create=create),
                            '{0}samples_itr_{1}.npz'.format(prefix, itr))

    def _itr_save_samples(self, itr, samples, prefix=''):
        Sample.save(self._itr_samples_file(itr, prefix=prefix), samples)

    ###################
    ### Run methods ###
    ###################

    def run(self):
        """
        for some iterations:
            gather data (i.e. samples)
                note: while gather data, record statistics to measure how well we're doing
            add new data to training data
            train neural network
        """
        try:
            ### find last model file
            for samples_start_itr in xrange(self._max_iter-1, -1, -1):
                sample_file = self._itr_samples_file(samples_start_itr, create=False)
                if os.path.exists(sample_file):
                    samples_start_itr += 1
                    break
            ### load initial dataset
            init_data_folder = params['probcoll'].get('init_data', None)
            if init_data_folder is not None:
                self._logger.info('Adding initial data')
                ext = os.path.splitext(self._itr_samples_file(0))[-1]
                fnames = [fname for fname in os.listdir(init_data_folder) if ext in fname]
                for fname in fnames:
                    self._logger.info('\t{0}'.format(fname))
                self.probcoll_model.add_data([os.path.join(init_data_folder, fname) for fname in fnames])

            ### if any data and haven't trained on it already, train on it
            if (samples_start_itr > 0 or init_data_folder is not None) and (samples_start_itr != self._max_iter):
                self._run_training(samples_start_itr)
            start_itr = samples_start_itr

            ### training loop
            for itr in xrange(start_itr, self._max_iter):
                self._run_itr(itr)
                self._run_training(itr)
                self.run_testing(itr)
                if not self.probcoll_model.sess.graph.finalized:
                    self.probcoll_model.sess.graph.finalize()
        finally:
            self.close()

    def _run_itr(self, itr):
        self._logger.info('')
        self._logger.info('=== Itr {0}'.format(itr))
        self._logger.info('')
        self._logger.info('Itr {0} running'.format(itr))
        self._run_rollout(itr)
        self._logger.info('Itr {0} adding data'.format(itr))
        self.probcoll_model.add_data([self._itr_samples_file(itr)])
        self._logger.info('Itr {0} training probability of collision'.format(itr))

    def close(self):
        for p in self._jobs:
            p.kill()
        self._agent.close()
        self.probcoll_model.close()
    
    def _run_training(self, itr):
        if itr >= params['probcoll']['training_start_iter']:
            if params['probcoll']['is_training']:
                if self._asynchronous:
                    self.probcoll_model.recover()
                    if not self._async_on:
                        self._async_training()
                        self._async_on = True
                else:
                    start = time.time()
                    self.probcoll_model.train()

    def run_testing(self, itr):
        if (itr == self._max_iter - 1 \
                or itr % params['probcoll']['testing']['itr_freq'] == 0): 
            self._logger.info('Itr {0} testing'.format(itr))
            if self._async_on:
                self._logger.debug('Recovering probcoll model')
                self.probcoll_model.recover()
            T = params['probcoll']['T']
            samples = []
#            reset_pos, reset_ori = self._agent.get_pos_ori() 
            for cond in xrange(params['probcoll']['testing']['num_rollout']):
                self._logger.info('\t\tTesting cond {0} itr {1}'.format(cond, itr))
                start = time.time()
                self._agent.reset()
#                self._agent.reset(hard_reset=True)
                _, sample_no_noise, t = self._agent.sample_policy(
                    self._mpc_policy,
                    T=T,
                    is_testing=True)

                if t + 1 < T:
                    self._logger.warning('\t\t\tCrashed at t={0}'.format(t))
                else:
                    self._logger.info('\t\t\tLasted for t={0}'.format(t))

                samples.append(sample_no_noise.match(slice(0, t + 1)))
                assert(samples[-1].isfinite())
                elapsed = time.time() - start
                self._logger.info('\t\t\tFinished cond {0} of testing ({1:.1f}s, {2:.3f}x real-time)'.format(
                    cond,
                    elapsed,
                    t*params['probcoll']['dt']/elapsed))
            self._itr_save_samples(itr, samples, prefix='testing_')
#            self._agent.reset(pos=reset_pos, ori=reset_ori)

    def _async_training(self):
        pass

    def _run_rollout(self, itr):
        if itr == 0:
            self._agent.reset()
        T = params['probcoll']['T']
        label_with_noise = params['probcoll']['label_with_noise']

        samples = []
        for cond in xrange(self._num_rollouts):
            self._agent.reset()
            self._logger.info('\t\tStarting cond {0} itr {1}'.format(cond, itr))
            start = time.time()
            sample_noise, sample_no_noise, t = self._agent.sample_policy(
                self._mpc_policy,
                T=T,
                time_step=self._time_step,
                only_noise=label_with_noise)
            if t + 1 < T:
                self._logger.warning('\t\t\tCrashed at t={0}'.format(t))
            else:
                self._logger.info('\t\t\tLasted for t={0}'.format(t))

            if label_with_noise:
                samples.append(sample_noise.match(slice(0, t + 1)))
            else:
                samples.append(sample_no_noise.match(slice(0, t + 1)))
            assert(samples[-1].isfinite())
            elapsed = time.time() - start
            self._logger.info('\t\t\tFinished cond {0} ({1:.1f}s, {2:.3f}x real-time)'.format(
                cond,
                elapsed,
                t*params['probcoll']['dt']/elapsed))
            self._time_step += t 
        self._itr_save_samples(itr, samples)
