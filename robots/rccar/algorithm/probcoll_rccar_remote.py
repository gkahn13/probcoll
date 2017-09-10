import os
import subprocess
import signal
import time
import numpy as np
import pickle as pkl

from general.algorithm.probcoll import Probcoll
from general.state_info.sample import Sample
from general.policy.random_policy import RandomPolicy
from general.policy.policy_cem import PolicyCem
from general.policy.policy_random_planning import PolicyRandomPlanning
from robots.rccar.agent.agent_rccar import AgentRCcar
from robots.rccar.algorithm.probcoll_model_rccar import ProbcollModelRCcar

from config import params

class ProbcollRCcarRemote(Probcoll):

    def __init__(self, save_dir=None, data_dir=None):
        if save_dir is None:
            save_dir = os.path.join(params['exp_dir_car'], params['exp_name'])
        Probcoll.__init__(self, save_dir=save_dir, data_dir=data_dir)
        self._start_data = os.path.join(self._save_dir, "start_data.pkl")

    def _setup(self):
        probcoll_params = params['probcoll']
        self._max_iter = probcoll_params['max_iter']
        self.agent = AgentRCcar(logger=self._logger)
        self._num_timesteps = params['probcoll']['num_timesteps']
        ### load prediction neural net
        if self._planner_type != 'random_policy':
            self.probcoll_model = ProbcollModelRCcar(save_dir=self._save_dir, data_dir=self._data_dir)

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
            if os.path.exists(self._start_data):
                with open(self._start_data, 'rb') as f:
                    start_itr = pkl.load(f)
            else:
                start_itr = 0
            if start_itr > 0:
                self._run_training(start_itr)
            self._time_step = self._num_timesteps * start_itr
            ### training loop
            for itr in range(start_itr, self._max_iter):
                with open(self._start_data, 'wb') as f:
                    pkl.dump(itr, f)
                self._run_rollout(itr)
                self._run_training(itr)
                self.run_testing(itr)
                if hasattr(self, 'probcoll_model') and not self.probcoll_model.sess.graph.finalized:
                    self.probcoll_model.sess.graph.finalize()
        finally:
            self.close()

    #########################
    ### Create controller ###
    #########################

    def _create_mpc(self):
        """ Must initialize MPC """
        self._logger.info('\t\t\tCreating MPC')
        if self._planner_type == 'random_policy':
            mpc_policy = RandomPolicy()
        elif self._planner_type == 'random':
            mpc_policy = PolicyRandomPlanning(self.probcoll_model, params['planning'])
        elif self._planner_type == 'cem':
            mpc_policy = PolicyCem(self.probcoll_model, params['planning'])
        else:
            raise NotImplementedError('planner_type {0} not implemented for rccar'.format(self._planner_type))

        return mpc_policy

    ###################
    ### Run methods ###
    ###################
    
    def close(self):
        self.agent.close()
    
    def _run_training(self, itr):
        if hasattr(self, 'probcoll_model'):
            self.probcoll_model.recover()
    
    def run_testing(self, itr):
        pass
