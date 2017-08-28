import os
import time
import numpy as np
import paramiko

from general.algorithm.probcoll import Probcoll
from general.algorithm.probcoll_model import ProbcollModel
from general.algorithm.probcoll_model_replay_buffer import ProbcollModelReplayBuffer
from general.state_info.sample import Sample
from general.policy.random_policy import RandomPolicy
from general.policy.policy_cem import PolicyCem
from general.policy.policy_random_planning import PolicyRandomPlanning
from robots.sim_rccar.agent.agent_sim_rccar import AgentSimRCcar

from config import params

class ProbcollRCcarPC(Probcoll):

    def __init__(self, save_dir=None, data_dir=None):
        Probcoll.__init__(self, save_dir=save_dir, data_dir=data_dir)

    def _setup(self):
        probcoll_params = params['probcoll']
        self._max_iter = probcoll_params['max_iter']
#        self.agent = AgentSimRCcar()
        self._num_timesteps = params['probcoll']['num_timesteps']
        ### load prediction neural net
        self.probcoll_model = ProbcollModel(save_dir=self._save_dir, data_dir=self._data_dir)
#        self.probcoll_model = ProbcollModelReplayBuffer(save_dir=self._save_dir, data_dir=self._data_dir)
        # TODO maybe move yaml
        self._ssh = paramiko.SSHClient()
        self._ssh.load_system_host_keys()
        self._ssh.connect(server, username=username, password=password)
        self._sftp = self._ssh.open_sftp()
        self._ssh.exec_command("python3", python_main_file, "probcoll", "rccar")

    ###################
    ### Run methods ###
    ###################
    def _run_itr(self, itr):
        self._run_rollout(itr)

    def close(self):
        for p in self._jobs:
            p.kill()
        self.probcoll_model.close()
        self._sftp.close()
        self._ssh.close()
    
    def _run_training(self, itr):
        if itr >= params['probcoll']['training_start_iter']:
            if params['probcoll']['is_training']:
                if self._asynchronous:
#                    self.probcoll_model.recover()
                    self._async_training()
                    self._async_on = True

    def _async_training(self):
        self.probcoll_model.train_loop()

    def _run_rollout(self, itr):
        for f in self._sftp.listdir(remote_data_dir):
            local_file = os.path.join(local_data_dir, f)
            remote_file = os.path.join(remote_data_dir, f)
            self._sftp.get(remote_file, local_file)
            self._sftp.remove(remote_file)

        for f in os.listdir(local_ckpt_dir):
            local_file = os.path.join(local_ckpt_dir, f)
            remote_file = os.path.join(remote_ckpt_dir, f)
            self._sftp.put(local_file, remote_file)

    def run_testing(self, itr):
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
            mpc_policy = PolicyRandomPlanning(self.probcoll_model, params['planning'])
        elif self._planner_type == 'cem':
            mpc_policy = PolicyCem(self.probcoll_model, params['planning'])
        else:
            raise NotImplementedError('planner_type {0} not implemented for rccar'.format(self._planner_type))

        return mpc_policy
