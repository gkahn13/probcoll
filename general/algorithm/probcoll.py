import abc

import os, pickle, time, shutil
import numpy as np
from general.algorithm.probcoll_model import ProbcollModel
from config import params

from general.policy.noise_models import ZeroNoise, GaussianNoise, UniformNoise, OUNoise, SmoothedGaussianNoise
from general.utility.logger import get_logger
from general.state_info.sample import Sample
class Probcoll:
    __metaclass__ = abc.ABCMeta

    def __init__(self, read_only=False):
        self._use_cp_cost = True
        self._planner_type = params['planning']['planner_type']
        self._read_only = read_only
        self._use_dynamics = True
        self._setup()
        self._logger = get_logger(
            self.__class__.__name__,
            'info',
            os.path.join(self._save_dir, 'dagger.txt'))
        self._mpc_policy = self._create_mpc()
        shutil.copy(params['yaml_path'], os.path.join(self._save_dir, '{0}.yaml'.format(params['exp_name'])))


    @abc.abstractmethod
    def _setup(self):
        ### load prediction neural net
        self._probcoll_model = None
        self._cost_probcoll = None
        self._max_iter = None
        self._world = None
        self._dynamics = None
        self._agent = None
        self._conditions = None

        raise NotImplementedError('Implement in subclass')

    #####################
    ### World methods ###
    #####################

    @abc.abstractmethod
    def _reset_world(self, itr, cond, rep):
        raise NotImplementedError('Implement in subclass')

    @abc.abstractmethod
    def _update_world(self, sample, t):
        raise NotImplementedError('Implement in subclass')

    @abc.abstractmethod
    def _is_good_rollout(self, sample, t):
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

    @property
    def _save_dir(self):
        return os.path.join(params['exp_dir'], params['exp_name'])

    def _itr_dir(self, itr, create=True):
        assert(type(itr) is int)
        dir = os.path.join(self._save_dir, 'itr{0}'.format(itr))
        if not create:
            return dir
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir

    def _itr_samples_file(self, itr, create=True):
        return os.path.join(self._itr_dir(itr, create=create),
                            'samples_itr_{0}.npz'.format(itr))

    def _itr_stats_file(self, itr, create=True):
        return os.path.join(self._itr_dir(itr, create=create),
                            'stats_itr_{0}.pkl'.format(itr))

    def _itr_save_worlds(self, itr, world_infos):
        fname = os.path.join(self._itr_dir(itr), 'worlds_itr_{0}.pkl'.format(itr))
        with open(fname, 'w') as f:
            pickle.dump(world_infos, f)

    def _itr_save_mpcs(self, itr, mpc_infos):
        fname = os.path.join(self._itr_dir(itr), 'mpcs_itr_{0}.pkl'.format(itr))
        with open(fname, 'w') as f:
            pickle.dump(mpc_infos, f)

    def _itr_save_samples(self, itr, samples):
        Sample.save(self._itr_samples_file(itr), samples)

    @property
    def _img_dir(self):
        dir = os.path.join(self._save_dir, 'images')
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir

    def _img_file(self, itr, cond, rep):
        return os.path.join(self._img_dir, 'itr{0:03d}_cond{1:03d}_rep{2:03d}.png'.format(itr, cond, rep))

    @abc.abstractmethod
    def _get_world_info(self):
        raise NotImplementedError('Implement in subclass')

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
            self._probcoll_model.add_data([os.path.join(init_data_folder, fname) for fname in fnames])

        ### if any data and haven't trained on it already, train on it
        if (samples_start_itr > 0 or init_data_folder is not None) and (samples_start_itr != self._max_iter):
            self._run_training()
        start_itr = samples_start_itr

        ### training loop
        for itr in xrange(start_itr, self._max_iter):
            self._run_itr(itr)
            self._run_training()
            if not self._probcoll_model.sess.graph.finalized:
                self._probcoll_model.sess.graph.finalize()
        self._close()

    def _run_itr(self, itr):
        self._logger.info('')
        self._logger.info('=== Itr {0}'.format(itr))
        self._logger.info('')
        self._logger.info('Itr {0} running'.format(itr))
        self._run_rollout(itr)
        self._logger.info('Itr {0} adding data'.format(itr))
        self._probcoll_model.add_data([self._itr_samples_file(itr)])
        self._logger.info('Itr {0} training probability of collision'.format(itr))

    def _close(self):
        self._probcoll_model.close()

    def _run_training(self):
        self._probcoll_model.train(reset=params['model']['reset_every_train'])

    def _run_rollout(self, itr):
        T = params['probcoll']['T']
        label_with_noise = params['probcoll']['label_with_noise']

        samples = []
        world_infos = []
        mpc_infos = []

        self._conditions.reset()
        for cond in xrange(self._conditions.length):
            rep = 0
            while rep < self._conditions.repeats:
                self._logger.info('\t\tStarting cond {0} rep {1} itr {2}'.format(cond, rep, itr))
                if (cond == 0 and rep == 0) or self._world.randomize:
                    self._reset_world(itr, cond, rep)

                x0 = self._conditions.get_cond(cond, rep=rep)
                sample_T = Sample(meta_data=params, T=T)
                sample_T.set_X(x0, t=0)
                
                # TODO memoryless vs not
#                mpc_policy = self._create_mpc(itr, x0)

                # For validation no noise
                if (cond >= self._conditions.length * \
                        params['model']['val_pct']) and \
                        (params['probcoll']['validation_noise']): 
                    control_noise = ZeroNoise()
                else:    
                    control_noise = self._create_control_noise() # create each time b/c may not be memoryless

                start = time.time()
                for t in xrange(T):
                    self._update_world(sample_T, t)

                    x0 = sample_T.get_X(t=t)

                    rollout = self._agent.sample_policy(x0, self._mpc_policy, noise=control_noise, T=1)

                    u = rollout.get_U(t=0)
                    o = rollout.get_O(t=0)

                    sample_T.set_U(u, t=t)
                    sample_T.set_O(o, t=t)
                    
                    if not self._use_dynamics:
                        sample_T.set_X(rollout.get_X(t=0), t=t)
                    
                    if self._world.is_collision(sample_T, t=t):
                        self._logger.warning('\t\t\tCrashed at t={0}'.format(t))
                        break

                    if self._use_dynamics:
                        if t < T-1:
                            x_tp1 = self._dynamics.evolve(x0, u)
                            sample_T.set_X(x_tp1, t=t+1)

                    if hasattr(self._mpc_policy, '_curr_traj'):
                        self._world.update_visualization(sample_T, self._mpc_policy._curr_traj, t)

                else:
                    self._logger.info('\t\t\tLasted for t={0}'.format(t))

                sample = sample_T.match(slice(0, t + 1))
                world_info = self._get_world_info()
                mpc_info = self._mpc_policy.get_info()

                if not self._is_good_rollout(sample, t):
                    self._logger.warning('\t\t\tNot good rollout. Repeating rollout.'.format(t))
                    continue

                samples.append(sample)
                world_infos.append(world_info)
                mpc_infos.append(mpc_info)

                assert(samples[-1].isfinite())
                elapsed = time.time() - start
                self._logger.info('\t\t\tFinished cond {0} rep {1} ({2:.1f}s, {3:.3f}x real-time)'.format(cond,
                                                                                                         rep,
                                                                                                         elapsed,
                                                                                                         t*params['dt']/elapsed))
                rep += 1

        self._itr_save_samples(itr, samples)
        self._itr_save_worlds(itr, world_infos)
        self._itr_save_mpcs(itr, mpc_infos)

        self._reset_world(itr, 0, 0) # leave the world as it was

    def _create_control_noise(self):
        cn_params = params['probcoll']['control_noise']

        if cn_params['type'] == 'zero':
            ControlNoiseClass = ZeroNoise
        elif cn_params['type'] == 'gaussian':
            ControlNoiseClass = GaussianNoise
        elif cn_params['type'] == 'uniform':
            ControlNoiseClass = UniformNoise
        elif cn_params['type'].lower() == 'ou':
            ControlNoiseClass = OUNoise
        elif cn_params['type'].lower() == 'smoothedgaussian':
            ControlNoiseClass = SmoothedGaussianNoise
        else:
            raise Exception('Control noise type {0} not valid'.format(cn_params['type']))

        return ControlNoiseClass(params, **cn_params[cn_params['type']])
