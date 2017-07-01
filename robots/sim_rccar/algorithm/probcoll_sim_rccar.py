import os
import subprocess
import signal
import time
import numpy as np

from general.tf.planning.planner_cem import PlannerCem
from general.algorithm.probcoll import Probcoll
from general.algorithm.probcoll_model import ProbcollModel
from general.policy.open_loop_policy import OpenLoopPolicy
from general.policy.random_policy import RandomPolicy
from general.state_info.conditions import Conditions
from general.state_info.sample import Sample
from robots.sim_rccar.tf.planning.planner_random_sim_rccar import PlannerRandomSimRCcar
from robots.sim_rccar.agent.agent_sim_rccar import AgentSimRCcar
from robots.sim_rccar.world.world_sim_rccar import WorldSimRCcar

from config import params

class ProbcollSimRCcar(Probcoll):

    def __init__(self, save_dir=None, data_dir=None):
        Probcoll.__init__(self, save_dir=save_dir, data_dir=data_dir)

    def _setup(self):
        probcoll_params = params['probcoll']
        world_params = params['world']
        self._max_iter = probcoll_params['max_iter']
        self._agent = AgentSimRCcar()
        self._world = WorldSimRCcar(self._agent, wp=world_params)
        self._num_rollouts = params['probcoll']['num_rollouts']
        # Keeps track of how many rollouts have been done
        self._rollout_num = 0
        ### load prediction neural net
        self.probcoll_model = ProbcollModel(save_dir=self._save_dir, data_dir=self._data_dir)
    
    ##########################
    ### Threaded Functions ###
    ##########################

    def _async_training(self):
        try:
            p = subprocess.Popen(
                ["python", "main.py", "train", "rccar", "--asynch"])
            self._jobs.append(p)
        except Exception as e:
            self._logger.warning('Error starting async training!')
            self._logger.warning(e)

    def close(self):
        for p in self._jobs:
            p.kill()
        self._agent.close()
        self.probcoll_model.close()
    
    ###################
    ### Run methods ###
    ###################

    def _run_rollout(self, itr):
        if itr == 0:
            self._agent.reset()
        T = params['probcoll']['T']
        label_with_noise = params['probcoll']['label_with_noise']

        samples = []
        mpc_infos = []
        for cond in xrange(self._num_rollouts):
            self._reset_world()
            self._logger.info('\t\tStarting cond {0} itr {1}'.format(cond, itr))
            start = time.time()
            sample_T = Sample(meta_data=params, T=T)
            for t in xrange(T):
                rollout, rollout_no_noise = self._agent.sample_policy(self._mpc_policy, self._rollout_num, T=1, time_step=t, only_noise=label_with_noise)
                
                o = rollout.get_O(t=0)
                x = rollout.get_X(t=0)
                if label_with_noise:
                    u = rollout.get_U(t=0)
                else:
                    u = rollout_no_noise.get_U(t=0)

                sample_T.set_U(u, t=t)
                sample_T.set_O(o, t=t)
                sample_T.set_X(x, t=t)
                
                if self._world.is_collision(sample_T, t=t):
                    self._logger.warning('\t\t\tCrashed at t={0}'.format(t))
                    break
            else:
                self._logger.info('\t\t\tLasted for t={0}'.format(t))

            sample = sample_T.match(slice(0, t + 1))
            mpc_info = self._mpc_policy.get_info()

            samples.append(sample)
            mpc_infos.append(mpc_info)

            assert(samples[-1].isfinite())
            elapsed = time.time() - start
            self._logger.info('\t\t\tFinished cond {0} ({1:.1f}s, {2:.3f}x real-time)'.format(
                cond,
                elapsed,
                t*params['probcoll']['dt']/elapsed))
            self._rollout_num += 1 
        self._itr_save_samples(itr, samples)
        self._itr_save_mpcs(itr, mpc_infos)
    
    def run_testing(self, itr):
        if (itr == self._max_iter - 1 \
                or itr % params['world']['testing']['itr_freq'] == 0): 
            self._logger.info('Itr {0} testing'.format(itr))
            if self._async_on:
                self._logger.debug('Recovering probcoll model')
                self.probcoll_model.recover()
            T = params['probcoll']['T']
            conditions = []
            if params['world']['testing'].get('position_ranges', None) is not None:
                ranges = params['world']['testing']['position_ranges']
                num_pos = params['world']['testing']['num_pos']
                if params['world']['testing']['range_type'] == 'random':
                    for _ in xrange(num_pos):
                        ran = ranges[np.random.randint(len(ranges))]
                        conditions.append(np.random.uniform(ran[0], ran[1]))
                elif params['world']['testing']['range_type'] == 'fix_spacing':
                    num_ran = len(ranges)
                    num_per_ran = num_pos//num_ran
                    for i in xrange(num_ran):
                        ran = ranges[i]
                        low = np.array(ran[0])
                        diff = np.array(ran[1]) - np.array(ran[0])
                        for j in xrange(num_per_ran):
                            val = diff * ((j + 0.0)/num_per_ran) + low
                            conditions.append(val) 
            elif params['world']['testing'].get('positions', None) is not None:
                conditions = params['world']['testing']['positions']
            samples = []
            reset_pos, reset_ori = self._agent.get_pos_ori() 
            for cond in xrange(len(conditions)):
                self._logger.info('\t\tTesting cond {0} itr {1}'.format(cond, itr))
                start = time.time()
                pos_ori = conditions[cond]
                pos = pos_ori[:3]
                ori = (0.0, 0.0, pos_ori[3])
                self._agent.reset(pos=pos, ori=ori)
                sample_T = Sample(meta_data=params, T=T)
                for t in xrange(T):
                    rollout, rollout_no_noise = self._agent.sample_policy(self._mpc_policy, 0., T=1, use_noise=False)
                    o = rollout.get_O(t=0)
                    x = rollout.get_X(t=0)
                    u = rollout_no_noise.get_U(t=0)

                    sample_T.set_U(u, t=t)
                    sample_T.set_O(o, t=t)
                    sample_T.set_X(x, t=t)
        
                    if self._world.is_collision(sample_T, t=t):
                        self._logger.warning('\t\t\tCrashed at t={0}'.format(t))
                        break

                else:
                    self._logger.info('\t\t\tLasted for t={0}'.format(t))

                sample = sample_T.match(slice(0, t + 1))
                samples.append(sample)
                assert(samples[-1].isfinite())
                elapsed = time.time() - start
                self._logger.info('\t\t\tFinished cond {0} of testing ({1:.1f}s, {2:.3f}x real-time)'.format(
                    cond,
                    elapsed,
                    t*params['probcoll']['dt']/elapsed))
            self._itr_save_samples(itr, samples, prefix='testing_')
            self._agent.reset(pos=reset_pos, ori=reset_ori)

    #####################
    ### World methods ###
    #####################

    def _reset_world(self):
        self._world.reset()

    def _update_world(self):
        pass
    
    def _is_good_rollout(self):
        pass
    
    #########################
    ### Create controller ###
    #########################

    def _create_mpc(self):
        """ Must initialize MPC """
        self._logger.info('\t\t\tCreating MPC')
        if self._planner_type == 'random_policy':
            mpc_policy = RandomPolicy()
        elif self._planner_type == 'random':
            planner = PlannerRandomSimRCcar(self.probcoll_model, params['planning'])
            mpc_policy = OpenLoopPolicy(planner)
        elif self._planner_type == 'cem':
            planner = PlannerCem(self.probcoll_model, params['planning'])
            mpc_policy = OpenLoopPolicy(planner)
        else:
            raise NotImplementedError('planner_type {0} not implemented for rccar'.format(self._planner_type))

        return mpc_policy

    ####################
    ### Info methods ###
    ####################

    def _get_world_info(self):
        return self._world.get_info()

