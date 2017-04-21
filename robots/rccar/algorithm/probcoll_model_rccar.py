import os, sys
import random

import numpy as np
import tensorflow as tf

from general.algorithm.probcoll_model import ProbcollModel

from config import params

class ProbcollModelRCcar(ProbcollModel):

    ####################
    ### Initializing ###
    ####################

    def __init__(self, read_only=False, finalize=True):
        dist_eps = params['O']['collision']['buffer']
        ProbcollModel.__init__(self, dist_eps, read_only=read_only, finalize=finalize)

    #############
    ### Files ###
    #############

    @property
    def _this_file(self):
        return os.path.abspath(__file__.replace('.pyc', '.py'))

    ############
    ### Data ###
    ############

# TODO figure out if I need this
#    def _modify_sample(self, sample):
#        """
#        In case you want to pre-process the sample before adding it
#        :return: Sample
#        """
#        # return ProbcollModel._modify_sample(self, sample)
#
#        ### move collision observation one time step earlier
#        if sample.get_O(t=-1, sub_obs='collision'):
#            try:
#                T_sample = len(sample.get_X())
#                t_backward = 0
#                new_sample = sample.match(slice(0, T_sample - t_backward))
#                new_sample.set_O([1.], t=-1, sub_obs='collision')
#                # t_forward = 2
#                # new_sample = sample.match(slice(t_forward, T_sample))
#                # new_sample.set_U(sample.get_U(t=slice(0, T_sample-t_forward)), t=slice(0, T_sample-t_forward))
#                # new_sample.set_X(sample.get_X(t=slice(0, T_sample-t_forward)), t=slice(0, T_sample-t_forward))
#                sample = new_sample
#            except:
#                self._logger.debug('Modify sample exception')
#
#        return [sample]

    #############
    ### Graph ###
    #############

    ################
    ### Training ###
    ################
