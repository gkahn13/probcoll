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
from robots.rccar.algorithm.probcoll_model_rccar import ProbcollModelRCcar

from config import params

class ProbcollRCcar(Probcoll):

    def __init__(self, server, username, password, reset=False, save_dir=None, data_dir=None):
        self._server = server
        self._username = username
        self._password = password
        self._reset_init = reset
        Probcoll.__init__(self, save_dir=save_dir, data_dir=data_dir)

    def _setup(self):
        probcoll_params = params['probcoll']
        self._max_iter = probcoll_params['max_iter']
        self._num_timesteps = params['probcoll']['num_timesteps']
        ### load prediction neural net
        self.probcoll_model = ProbcollModelRCcar(save_dir=self._save_dir, data_dir=self._data_dir, trainable=True)
        self._remote_save_dir = os.path.join(params['exp_dir_car'], params['exp_name'])
        self._remote_samples = os.path.join(self._remote_save_dir, "samples")
        self._remote_ckpt = os.path.join(self._remote_save_dir, "model_checkpoints")
        self._ssh = paramiko.SSHClient()
        self._ssh.load_system_host_keys()
        self._ssh.connect(self._server, username=self._username, password=self._password)
        self._sftp = self._ssh.open_sftp()
        self._logger.info("SSH connected")
        # Move yaml
        local_yaml = os.path.join(self._save_dir, '{0}.yaml'.format(params['exp_name']))
        remote_yaml = os.path.join(self._remote_save_dir, '{0}.yaml'.format(params['exp_name']))
        self._ssh.exec_command("mkdir {0} -p".format(self._remote_samples))
        self._ssh.exec_command("mkdir {0} -p".format(self._remote_ckpt))
        self._logger.info("Added experiments directory remotely")
        time.sleep(0.1)
        self._sftp.put(local_yaml, remote_yaml)
        self._logger.info("Added yaml remotely")
        input("Press Enter once you start probcoll on rccar")

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
            for itr in range(self._max_iter):
                sample_file = self._itr_samples_file(itr, create=False)
                if os.path.exists(sample_file):
                    self.probcoll_model.add_data([sample_file])
                else:
                    break

            samples_start_itr = itr
            
            ### if any data and haven't trained on it already, train on it
            self.probcoll_model.recover()
            if samples_start_itr > 0: 
                self._run_training(samples_start_itr, reset=self._reset_init)
            elif init_data_folder is not None:
                self._run_training(samples_start_itr)
            
            self._itr = samples_start_itr
            self._time_step = self._num_timesteps * self._itr
            ### training loop
            while self._itr < self._max_iter:
                self._get_data()
                self._run_training(self._itr)
                if not params['probcoll']['save_O']:
                    self._itr_remove_O_from_samples(itr)
                if not self.probcoll_model.sess.graph.finalized:
                    self.probcoll_model.sess.graph.finalize()
        finally:
            self.close()
   
    def _run_itr(self, itr):
        pass

    def _run_rollout(self, itr):
        pass

    def _run_training(self, itr, reset=False):
        if itr >= params['probcoll']['training_start_iter']:
            if params['probcoll']['is_training']:
                # TODO maybe add different training
                self._async_training(reset=reset)

    def _async_training(self, reset=False):
        self.probcoll_model.train(reset=reset)
        try:
            for f in os.listdir(self.probcoll_model.checkpoints_dir):
                local_file = os.path.join(self.probcoll_model.checkpoints_dir, f)
                remote_file = os.path.join(self._remote_ckpt, f)
                self._sftp.put(local_file, remote_file)
        except:
            self._logger.warning("Checkpoint file not updated")

    def run_testing(self, itr):
        pass

    def _get_data(self):
        for f in self._sftp.listdir(self._remote_samples):
            remote_file = os.path.join(self._remote_samples, f)
            self._logger.info('Found sample itr {0}'.format(self._itr+1))
            while True:
                try:
                    local_file = self._itr_samples_file(self._itr)
                    self._sftp.get(remote_file, local_file)
                    Sample.load(local_file)
                    break
                except Exception as e:
                    print(e)
            self.probcoll_model.add_data([local_file])
            self._sftp.remove(remote_file)
            self._itr += 1

    #########################
    ### Create controller ###
    #########################

    def _create_mpc(self):
        pass

    ###############
    ### Closing ###
    ###############

    def close(self):
        for p in self._jobs:
            p.kill()
        self.probcoll_model.close()
        self._sftp.close()
        self._ssh.close()
    
