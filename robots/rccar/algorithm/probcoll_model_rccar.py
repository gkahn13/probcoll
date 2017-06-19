import os, sys
import random
import numpy as np
import tensorflow as tf
import threading

from general.algorithm.probcoll_model import ProbcollModel

from config import params

class ProbcollModelRCcar(ProbcollModel):

    ####################
    ### Initializing ###
    ####################

    #############
    ### Files ###
    #############

    @property
    def _this_file(self):
        return os.path.abspath(__file__.replace('.pyc', '.py'))

    ############
    ### Data ###
    ############

    #############
    ### Graph ###
    #############

    ################
    ### Training ###
    ################

    def async_training(self):
        t = threading.Thread(
            target=ProbcollModelRCcar.async_train_func,
            args=(self,))
        t.daemon = True
        self.threads.append(t)
        self.coord.register_thread(t)
        t.start()

    def async_train_func(self):
        self._logger.info("Started asynchronous training!")
        try:
            while (True):
                self.train()
        finally:
            self._logger.info("Ending asynchronous training!")
