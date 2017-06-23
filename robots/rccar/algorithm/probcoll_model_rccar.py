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
