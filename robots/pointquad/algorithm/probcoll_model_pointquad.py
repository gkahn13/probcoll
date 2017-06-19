import os
import random
import sys
import numpy as np
import tensorflow as tf

from general.algorithm.probcoll_model import ProbcollModel

from config import params

class ProbcollModelPointquad(ProbcollModel):

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

    def _modify_sample(self, sample):
        """
        In case you want to pre-process the sample before adding it
        :return: Sample
        """
        ### move collision observation one time step earlier
        if sample.get_O(t=-1, sub_obs='collision'):
            try:
                T_sample = len(sample.get_X())
                new_sample = sample.match(slice(0, T_sample-1))
                new_sample.set_O([1.], t=-1, sub_obs='collision')
                sample = new_sample
            except:
                self._logger.debug('Modify sample exception')

        return [sample]

    #############
    ### Graph ###
    #############

    ################
    ### Training ###
    ################

    def _create_input(self, X, U, O):
        return ProbcollModel._create_input(self, X, U, O)

    def _create_output(self, output):
        return ProbcollModel._create_output(self, output).astype(int)

