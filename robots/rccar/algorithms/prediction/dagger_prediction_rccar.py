import os

import rospy
import std_msgs.msg as std_msgs
from general.algorithm.probcoll import Probcoll
from rll_quadrotor.policy.cem_mpc_policy import CEMMPCPolicy
from robots.rccar.algorithm.cost_probcoll_rccar import CostProbcollRCcar
from robots.rccar.algorithm.probcoll_model_rccar import ProbcollModelRCcar

import general.ros.ros_utils as ros_utils
from config import params
from general.state_info.conditions import Conditions
from general.state_info.sample import Sample
from robots.rccar.agent.agent_rccar import AgentRCcar
from robots.rccar.dynamics.dynamics_rccar import DynamicsRCcar
from robots.rccar.policy.lattice_mpc_policy_rccar import LatticeMPCPolicyRCcar
from robots.rccar.policy.primitives_mpc_policy_rccar import PrimitivesMPCPolicyRCcar
from robots.rccar.policy.straight_policy_rccar import StraightPolicyRCcar
from robots.rccar.policy.teleop_mpc_policy_rccar import TeleopMPCPolicyRCcar
from robots.rccar.traj_opt.ilqr.cost.cost_velocity_rccar import cost_velocity_rccar
from robots.rccar.traj_opt.traj_opt_rccar import TrajoptRCcar
from robots.rccar.world.world_rccar import WorldRCcar

class ProbcollRCcar(Probcoll):

    def __init__(self, read_only=False):
        Probcoll.__init__(self, read_only=read_only)

    def _setup(self):
        rospy.init_node('ProbcollRCcar', anonymous=True)

        pred_dagger_params = params['prediction']['dagger']
        world_params = params['world']
        cond_params = pred_dagger_params['conditions']
        cp_params = pred_dagger_params['cost_probcoll']

        self._max_iter = pred_dagger_params['max_iter']
        self._dynamics = DynamicsRCcar()
        self._agent = AgentRCcar(self._dynamics)
        self._world = WorldRCcar(self._agent, self._bag_file, wp=world_params)
        self._trajopt = TrajoptRCcar(self._dynamics, self._world, self._agent)
        self._conditions = Conditions(cond_params=cond_params)

        assert(self._world.randomize)

        ### load prediction neural net
        self._probcoll_model = ProbcollModelRCcar(read_only=self._read_only)

        self._cost_probcoll = CostProbcollRCcar(self._probcoll_model,
                                           weight=float(cp_params['weight']),
                                           eval_cost=cp_params['eval_cost'],
                                           pre_activation=cp_params['pre_activation'])

        rccar_topics = params['rccar']['topics']
        self.coll_callback = ros_utils.RosCallbackEmpty(rccar_topics['collision'], std_msgs.Empty)
        self.good_rollout_callback = ros_utils.RosCallbackEmpty(rccar_topics['good_rollout'], std_msgs.Empty)
        self.bad_rollout_callback = ros_utils.RosCallbackEmpty(rccar_topics['bad_rollout'], std_msgs.Empty)

    ####################
    ### Save methods ###
    ####################

    def _bag_file(self, itr, cond, rep, create=True):
        return os.path.join(self._itr_dir(itr, create=create), 'bagfile_itr{0}_cond{1}_rep{2}.bag'.format(itr, cond, rep))

    def _itr_save_worlds(self, itr, world_infos):
        pass

    #####################
    ### World methods ###
    #####################

    def _reset_world(self, itr, cond, rep):
        if cond == 0 and rep == 0:
            self._logger.info('Press A or B to start')
            self._ros_is_good_rollout()
        back_up = self.coll_callback.get() is not None # only back up if experienced a crash
        self._world.reset(back_up, itr=itr, cond=cond, rep=rep)

    def _update_world(self, sample, t):
        return

    def _is_good_rollout(self, sample, t):
        self._agent.execute_control(None) # stop the car
        self._logger.info('Is good rollout? (A for yes, B for no)')
        return self._ros_is_good_rollout()

    def _ros_is_good_rollout(self):
        self.good_rollout_callback.get()
        self.bad_rollout_callback.get()
        while not rospy.is_shutdown():
            good_rollout = self.good_rollout_callback.get()
            bad_rollout = self.bad_rollout_callback.get()
            if good_rollout and not bad_rollout:
                return True
            elif bad_rollout and not good_rollout:
                return False
            rospy.sleep(0.1)

    #########################
    ### Create controller ###
    #########################

    def _create_mpc(self, itr, x0):
        """ Must initialize MPC """
        sample0 = Sample(meta_data=params, T=1)
        sample0.set_X(x0, t=0)
        self._update_world(sample0, 0)

        self._logger.info('\t\t\tCreating MPC')

        if self._planner_type == 'primitives':
            additional_costs = []
            mpc_policy = PrimitivesMPCPolicyRCcar(self._trajopt,
                                                  self._cost_probcoll,
                                                  additional_costs=additional_costs,
                                                  meta_data=params,
                                                  use_threads=False,
                                                  plot=True,
                                                  epsilon_greedy=params['prediction']['dagger']['epsilon_greedy'])
        elif self._planner_type == 'cem':
            costs = [self._cost_probcoll,
                     cost_velocity_rccar(params['mpc']['H'],
                                         params['trajopt']['cost_velocity']['u_des'],
                                         params['trajopt']['cost_velocity']['u_weights'],
                                         weight_scale=1.0)]
            mpc_policy = CEMMPCPolicy(self._world,
                                      self._dynamics,
                                      costs,
                                      meta_data=params)
        elif self._planner_type == 'straight':
            mpc_policy = StraightPolicyRCcar(meta_data=params)
        elif self._planner_type == 'teleop':
            mpc_policy = TeleopMPCPolicyRCcar(meta_data=params)
        elif self._planner_type == 'lattice':
            additional_costs = []
            mpc_policy = LatticeMPCPolicyRCcar(self._trajopt,
                                               self._cost_probcoll,
                                               additional_costs=additional_costs,
                                               meta_data=params,
                                               use_threads=False,
                                               plot=True,
                                               epsilon_greedy=params['prediction']['dagger']['epsilon_greedy'])
        else:
            raise NotImplementedError('planner_type {0} not implemented for rccar'.format(self._planner_type))

        return mpc_policy

    ####################
    ### Info methods ###
    ####################

    def _get_world_info(self):
        ### just returns empty dict, but function call terminates bag recording
        return self._world.get_info()

