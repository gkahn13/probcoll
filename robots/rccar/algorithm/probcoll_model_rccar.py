import abc
import os
import tensorflow as tf

from general.algorithm.probcoll_model_replay_buffer import ProbcollModelReplayBuffer 

class ProbcollModelRCcar(ProbcollModelReplayBuffer):
    
    #############################
    ### Load/save/reset/close ###
    #############################

    def get_latest_checkpoint(self):
        ckpt_file = os.path.join(self._checkpoints_dir, "0.ckpt")
        check_file = os.path.join(self._checkpoints_dir, "0.ckpt.index")
        if os.path.exists(check_file):
            return ckpt_file
        else:
            return None

    def recover(self):
        latest_file = self.get_latest_checkpoint()
        if latest_file is not None:
            while os.path.exists(latest_file):
                try:
                    self.load(latest_file)
                    break
                except Exception as e:
                    print(e)
        else:
            self._logger.info("Could not find checkpoint file")
        self._logger.info("Found checkpoint file")
