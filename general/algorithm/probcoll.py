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
        self.policy = self._create_mpc()
        new_yaml_path = os.path.join(self._save_dir, '{0}.yaml'.format(params['exp_name']))
        if not os.path.exists(new_yaml_path):
            with open(new_yaml_path, 'w') as f:
                f.write(params.pop('yaml_txt'))

    @abc.abstractmethod
    def _setup(self):
        ### load prediction neural net
        self.probcoll_model = None
        self._asynchprobcoll_model = None
        self._max_iter = None
        self.agent = None
        self.agent_testing = None
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

    def _itr_remove_O_from_samples(self, itr):
        for prefix in ('', 'testing_'):
            try:
                samples = Sample.load(self._itr_samples_file(itr, prefix=prefix, create=False))
                Sample.save(self._itr_samples_file(itr, prefix=prefix), samples, save_O=False)
            except:
                pass

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
            samples_start_itr = 0
            # Keeps track of how many rollouts have been done
            self._time_step = 0
            for samples_start_itr in range(self._max_iter-1, -1, -1):
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
            self._time_step = self._num_timesteps * start_itr
            ### training loop
            for itr in range(start_itr, self._max_iter):
                self._run_itr(itr)
                self._run_training(itr)
                self.run_testing(itr)
                if not params['probcoll']['save_O']:
                    self._itr_remove_O_from_samples(itr)
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
        self.agent.close()
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

    def _async_training(self):
        pass

    def _run_rollout(self, itr):
        T = params['probcoll']['T']
        label_with_noise = params['probcoll']['label_with_noise']
        self._logger.info('\t\tStarting itr {0}'.format(itr))
        samples = []
        iteration_steps = 0
        start = time.time()
        while iteration_steps < self._num_timesteps:
            max_T = min(T, self._num_timesteps - iteration_steps) 
            sample_noise, sample_no_noise, t = self.agent.sample_policy(
                self.policy,
                T=max_T,
                time_step=self._time_step,
                only_noise=label_with_noise)
            iteration_steps += t + 1
            self._time_step += t + 1 
            if label_with_noise:
                samples.append(sample_noise.match(slice(0, t + 1)))
            else:
                samples.append(sample_no_noise.match(slice(0, t + 1)))
            assert(samples[-1].isfinite())
            if samples[-1].get_O(t=t, sub_obs='collision'):
                self._logger.warning('\t\t\tCrashed at t={0}'.format(t))
            else:
                self._logger.info('\t\t\tLasted for t={0}'.format(t))

        assert(self._num_timesteps == iteration_steps)
        elapsed = time.time() - start
        self._logger.info('\t\t\tFinished itr {0} ({1:.1f}s, {2:.3f}x real-time)'.format(
            itr,
            elapsed,
            self._num_timesteps*params['probcoll']['dt']/elapsed))
        self._itr_save_samples(itr, samples)

    def run_testing(self, itr):
        if itr >= params['probcoll']['training_start_iter']\
                and ((itr == self._max_iter - 1 or itr == 0 \
                or (itr - 1) % params['probcoll']['testing']['itr_freq'] == 0)): 
            self._logger.info('Itr {0} testing'.format(itr))
            if self._async_on:
                self._logger.debug('Recovering probcoll model')
                self.probcoll_model.recover()
            T = params['probcoll']['T']
            samples = []
            self.agent.reset()
            total_time = 0
            start = time.time()
            for cond in range(params['probcoll']['testing']['num_rollout']):
                self.agent_testing.reset(hard_reset=True, is_testing=True)
                _, sample_no_noise, t = self.agent_testing.sample_policy(
                    self.policy,
                    T=T,
                    is_testing=True)
                total_time += t
                if sample_no_noise.get_O(t=t, sub_obs='collision'):
                    self._logger.warning('\t\t\tCrashed at t={0}'.format(t))
                else:
                    self._logger.info('\t\t\tLasted for t={0}'.format(t))

                samples.append(sample_no_noise.match(slice(0, t + 1)))
                assert(samples[-1].isfinite())
            elapsed = time.time() - start
            self._logger.info('\t\t\tFinished testing itr {0} ({1:.1f}s, {2:.3f}x real-time)'.format(
                itr,
                elapsed,
                total_time*params['probcoll']['dt']/elapsed))
            self._itr_save_samples(itr, samples, prefix='testing_')

