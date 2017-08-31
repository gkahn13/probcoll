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

    def __init__(self, server, username, password, save_dir=None, data_dir=None):
        self._server = server
        self._username = username
        self._password = password
        Probcoll.__init__(self, save_dir=save_dir, data_dir=data_dir)

    def _setup(self):
        probcoll_params = params['probcoll']
        self._max_iter = probcoll_params['max_iter']
        self._num_timesteps = params['probcoll']['num_timesteps']
        ### load prediction neural net
        self.probcoll_model = ProbcollModel(save_dir=self._save_dir, data_dir=self._data_dir)
#        self.probcoll_model = ProbcollModelReplayBuffer(save_dir=self._save_dir, data_dir=self._data_dir)
        self._remote_save_dir = os.path.join(params['exp_dir_car'], params['exp_name'])
        self._remote_samples = os.path.join(self._remote_save_dir, "samples")
        self._remote_ckpt = os.path.join(self._remote_save_dir, "model_checkpoints")
        self._remote_main = params['remote_main']
        self._ssh = paramiko.SSHClient()
        self._ssh.load_system_host_keys()
        self._ssh.connect(self._server, username=self._username, password=self._password)
        self._sftp = self._ssh.open_sftp()
        # Move yaml
        local_yaml = os.path.join(self._save_dir, '{0}.yaml'.format(params['exp_name']))
        remote_yaml = os.path.join(self._remote_save_dir, '{0}.yaml'.format(params['exp_name']))
        self._ssh.exec_command(["mkdir", "experiments/rccar", "-p"])
        self._sftp.put(local_yaml, remote_yaml)
        self._ssh.exec_command(["python3", self._remote_main, "probcoll", "rccar", "-yaml", remote_yaml])

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
            self._time_step = 0
            ### load initial dataset
            init_data_folder = params['probcoll'].get('init_data', None)
            if init_data_folder is not None:
                self._logger.info('Adding initial data')
                ext = os.path.splitext(self._itr_samples_file(0))[-1]
                fnames = [fname for fname in os.listdir(init_data_folder) if ext in fname]
                for fname in fnames:
                    self._logger.info('\t{0}'.format(fname))
                self.probcoll_model.add_data([os.path.join(init_data_folder, fname) for fname in fnames])
            # Keeps track of how many rollouts have been done
            for samples_start_itr in range(self._max_iter-1, -1, -1):
                sample_file = self._itr_samples_file(samples_start_itr, create=False)
                if os.path.exists(sample_file):
                    samples_start_itr += 1
                    break
            ### if any data and haven't trained on it already, train on it
            if (samples_start_itr > 0 or init_data_folder is not None) and (samples_start_itr != self._max_iter):
                self._run_training(samples_start_itr)
            self._itr = samples_start_itr
            self._time_step = self._num_timesteps * start_itr
            ### training loop
            while self._itr < self._max_iter:
                self._run_itr(self._itr)
                self._run_training(self._itr)
                if not params['probcoll']['save_O']:
                    self._itr_remove_O_from_samples(itr)
                if not self.probcoll_model.sess.graph.finalized:
                    self.probcoll_model.sess.graph.finalize()
        finally:
            self.close()
    
    def _run_itr(self, itr):
        self._run_rollout(itr)
    
    def _run_rollout(self, itr):
        for f in self._sftp.listdir(self._remote_samples):
            remote_file = os.path.join(self._remote_samples, f)
            local_file = self._itr_samples_file(itr):
            self._sftp.get(remote_file, local_file)
            self._sftp.remove(remote_file)
            self._itr += 1
    
    def _run_training(self, itr):
        if itr >= params['probcoll']['training_start_iter']:
            if params['probcoll']['is_training']:
                if self._asynchronous:
#                    self.probcoll_model.recover()
                    self._async_training()
                    self._async_on = True

    def _async_training(self):
#        self.probcoll_model.train_loop()
        self.probcoll_model.train()
        local_file = self.probcoll_model.get_latest_checkpoint()
        remote_file = os.path.join(self._remote_ckpt, os.path.split(local_file)[-1])
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

    ###############
    ### Closing ###
    ###############

    def close(self):
        for p in self._jobs:
            p.kill()
        self.probcoll_model.close()
        self._sftp.close()
        self._ssh.close()
    
