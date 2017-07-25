import numpy as np
import openravepy as rave
import rospy
import os
from config import params
from general.state_info.conditions import Conditions
from general.state_info.sample import Sample
from robots.bebop2d.agent.agent_bebop2d import AgentBebop2d
from robots.bebop2d.dynamics.dynamics_bebop2d import DynamicsBebop2d

from general.algorithm.probcoll import Probcoll
from general.algorithm.probcoll_model import ProbcollModel
from general.policy.random_policy import RandomPolicy
from robots.bebop2d.policy.policy_random_planning_bebop2d import PolicyRandomPlanningBebop2d
from robots.bebop2d.agent.agent_bebop2d import AgentBebop2d

import std_msgs.msg as std_msgs
import general.ros.ros_utils as ros_utils
import IPython
class ProbcollBebop2d(Probcoll):


    def __init__(self, save_dir=None, data_dir=None):
        Probcoll.__init__(self, save_dir=save_dir, data_dir=data_dir)

    def _setup(self):
        probcoll_params = params['probcoll']
        self._max_iter = probcoll_params['max_iter']
        self._num_rollouts = params['probcoll']['num_rollouts']
        # Keeps track of how many rollouts have been done
        self._rollout_num = 0
        ### load prediction neural net
        self.probcoll_model = ProbcollModel(save_dir=self._save_dir, data_dir=self._data_dir)
        self._agent = AgentBebop2d()

    def _bag_file(self, itr, cond, rep, create=True):
        return os.path.join(self._itr_dir(itr, create=create), 'bagfile_itr{0}_cond{1}_rep{2}.bag'.format(itr, cond, rep))

    def _create_mpc(self):
        """ Must initialize MPC """
#        self._logger.info('\t\t\tCreating MPC')
        if self._planner_type == 'random_policy':
            mpc_policy = RandomPolicy()
        elif self._planner_type == 'random':
            mpc_policy = PolicyRandomPlanningBebop2d(self.probcoll_model, params['planning'])
        else:
            raise NotImplementedError('planner_type {0} not implemented for bebop2d'.format(self._planner_type))
        return mpc_policy
