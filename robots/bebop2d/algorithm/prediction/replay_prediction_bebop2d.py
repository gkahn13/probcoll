import copy
import os

import numpy as np
from robots.bebop2d.algorithm.probcoll_bebop2d import ProbcollBebop2d

from config import params, load_params
from general.algorithm.probcoll_model import ProbcollModel
from general.state_info.conditions import Conditions
from general.state_info.sample import Sample
from robots.bebop2d.policy.primitives_mpc_policy_bebop2d import PrimitivesMPCPolicyBebop2d
from robots.bebop2d.policy.teleop_mpc_policy_bebop2d import TeleopMPCPolicyBebop2d


class ReplayPredictionBebop2d(ProbcollBebop2d):

    def __init__(self):
        np.random.seed(0)

        self._save_dir = os.path.join(params['exp_dir'], params['exp_name'])

        yamls = [fname for fname in os.listdir(self._save_dir) if '.yaml' in fname and '~' not in fname]
        assert (len(yamls) == 1)
        yaml_path = os.path.join(self._save_dir, yamls[0])
        load_params(yaml_path)
        params['yaml_path'] = yaml_path

        ### turn off noise
        pred_dagger_params = params['prediction']['dagger']
        pred_dagger_params['control_noise']['type'] = 'zero'
        pred_dagger_params['epsilon_greedy'] = None

        ProbcollBebop2d.__init__(self, read_only=True)

        cond_params = copy.deepcopy(pred_dagger_params['conditions'])
        cond_params['repeats'] = 100

        self._conditions = Conditions(cond_params=cond_params)

        self._cost_probcoll_init = None

    #############
    ### Files ###
    #############

    def _itr_dir(self, itr, create=True):
        assert(type(itr) is int)
        dir = os.path.join(self._save_dir, 'replay/itr{0}'.format(itr))
        if not create:
            return dir
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir

    #####################
    ### World methods ###
    #####################

    def _reset_world(self, itr, cond, rep):
        self._agent.execute_control(None) # stop bebop
        self.bad_rollout_callback.get() # to clear it
        self._world.reset(itr=itr, cond=cond, rep=rep, record=False)

    #########################
    ### Create controller ###
    #########################

    def _create_mpc(self, itr, x0):
        """ Must initialize MPC """
        sample0 = Sample(meta_data=params, T=1)
        sample0.set_X(x0, t=0)
        self._update_world(sample0, 0)

        self._logger.info('\t\t\tCreating MPC')

        if self._planner_type == 'primitives':
            additional_costs = []
            mpc_policy = PrimitivesMPCPolicyBebop2d(self._trajopt,
                                                    self._cost_probcoll,
                                                    additional_costs=additional_costs,
                                                    meta_data=params,
                                                    use_threads=False,
                                                    plot=True,
                                                    epsilon_greedy=params['prediction']['dagger']['epsilon_greedy'])
        elif self._planner_type == 'teleop':
            mpc_policy = TeleopMPCPolicyBebop2d(params)
        else:
            raise NotImplementedError('planner_type {0} not implemented for bebop2d'.format(self._planner_type))

        return mpc_policy

    ##############
    ### Replay ###
    ##############

    def replay(self):
        itr = 0
        while self.replay_itr(itr):
            itr += 1

    def replay_itr(self, itr):
        model_file = self._itr_model_file(itr).replace('replay/', '')
        if not ProbcollModel.checkpoint_exists(model_file):
            return False

        self._logger.info('Replaying itr {0}'.format(itr))
        self._probcoll_model.load(model_file=model_file)
        self._run_itr(itr)
        # self._run_itr_analyze(itr)

        return True

    def _run_itr_analyze(self, itr):
        T = 1
        samples = []
        world_infos = []
        mpc_infos = []

        self._conditions.reset()
        for cond in xrange(self._conditions.length):
            for rep in xrange(self._conditions.repeats):
                self._logger.info('\t\tStarting cond {0} rep {1}'.format(cond, rep))
                if (cond == 0 and rep == 0) or self._world.randomize:
                    self._reset_world(itr, cond, rep)

                x0 = self._conditions.get_cond(cond, rep=rep)
                sample_T = Sample(meta_data=params, T=T)
                sample_T.set_X(x0, t=0)

                mpc_policy = self._create_mpc(itr, x0)
                control_noise = self._create_control_noise() # create each time b/c may not be memoryless

                for t in xrange(T):
                    self._update_world(sample_T, t)

                    x0 = sample_T.get_X(t=t)

                    rollout = self._agent.sample_policy(x0, mpc_policy, noise=control_noise, T=1)

                    u = rollout.get_U(t=0)
                    o = rollout.get_O(t=0)
                    u_label = u

                    sample_T.set_U(u_label, t=t)
                    sample_T.set_O(o, t=t)

                    if hasattr(mpc_policy, '_curr_traj'):
                        self._world.update_visualization(sample_T, mpc_policy._curr_traj, t)

                else:
                    self._logger.info('\t\t\tLasted for t={0}'.format(t))

                samples.append(sample_T)
                world_infos.append(self._get_world_info())
                mpc_infos.append(mpc_policy.get_info())

                assert(samples[-1].isfinite())
                rep += 1

        self._itr_save_samples(itr, samples)
        self._itr_save_worlds(itr, world_infos)
        self._itr_save_mpcs(itr, mpc_infos)

