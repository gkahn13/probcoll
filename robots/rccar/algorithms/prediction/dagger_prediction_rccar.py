import os

import rospy
import std_msgs.msg as std_msgs

from general.algorithms.prediction.dagger_prediction import DaggerPrediction
from general.traj_opt.conditions import Conditions
import general.ros.ros_utils as ros_utils

from robots.rccar.algorithms.prediction.prediction_model_rccar import PredictionModelRCcar
from robots.rccar.algorithms.prediction.cost_prediction_rccar import CostPredictionRCcar

from robots.rccar.dynamics.dynamics_rccar import DynamicsRCcar
from robots.rccar.world.world_rccar import WorldRCcar
from robots.rccar.agent.agent_rccar import AgentRCcar
from robots.rccar.traj_opt.traj_opt_rccar import TrajoptRCcar
from robots.rccar.policy.primitives_mpc_policy_rccar import PrimitivesMPCPolicyRCcar
from robots.rccar.policy.straight_policy_rccar import StraightPolicyRCcar
from robots.rccar.policy.teleop_mpc_policy_rccar import TeleopMPCPolicyRCcar
from robots.rccar.policy.lattice_mpc_policy_rccar import LatticeMPCPolicyRCcar
from robots.rccar.traj_opt.ilqr.cost.cost_velocity_rccar import cost_velocity_rccar

from rll_quadrotor.state_info.sample import Sample
from rll_quadrotor.policy.cem_mpc_policy import CEMMPCPolicy

from config import params

class DaggerPredictionRCcar(DaggerPrediction):

    def __init__(self, read_only=False):
        DaggerPrediction.__init__(self, read_only=read_only)

    def _setup(self):
        rospy.init_node('DaggerPredictionRCcar', anonymous=True)

        pred_dagger_params = params['prediction']['dagger']
        world_params = params['world']
        cond_params = pred_dagger_params['conditions']
        cp_params = pred_dagger_params['cost_prediction']

        self.max_iter = pred_dagger_params['max_iter']
        self.dynamics = DynamicsRCcar()
        self.agent = AgentRCcar(self.dynamics)
        self.world = WorldRCcar(self.agent, self._bag_file, wp=world_params)
        self.trajopt = TrajoptRCcar(self.dynamics, self.world, self.agent)
        self.conditions = Conditions(cond_params=cond_params)

        assert(self.world.randomize)

        ### load prediction neural net
        self.bootstrap = PredictionModelRCcar(read_only=self.read_only)

        self.cost_cp = CostPredictionRCcar(self.bootstrap,
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
            self.logger.info('Press A or B to start')
            self._ros_is_good_rollout()
        back_up = self.coll_callback.get() is not None # only back up if experienced a crash
        self.world.reset(back_up, itr=itr, cond=cond, rep=rep)

    def _update_world(self, sample, t):
        return

    def _is_good_rollout(self, sample, t):
        self.agent.execute_control(None) # stop the car        
        self.logger.info('Is good rollout? (A for yes, B for no)')
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

        self.logger.info('\t\t\tCreating MPC')

        if self.planner_type == 'primitives':
            additional_costs = []
            mpc_policy = PrimitivesMPCPolicyRCcar(self.trajopt,
                                                  self.cost_cp,
                                                  additional_costs=additional_costs,
                                                  meta_data=params,
                                                  use_threads=False,
                                                  plot=True,
                                                  epsilon_greedy=params['prediction']['dagger']['epsilon_greedy'])
        elif self.planner_type == 'cem':
            costs = [self.cost_cp,
                     cost_velocity_rccar(params['mpc']['H'],
                                         params['trajopt']['cost_velocity']['u_des'],
                                         params['trajopt']['cost_velocity']['u_weights'],
                                         weight_scale=1.0)]
            mpc_policy = CEMMPCPolicy(self.world,
                                      self.dynamics,
                                      costs,
                                      meta_data=params)
        elif self.planner_type == 'straight':
            mpc_policy = StraightPolicyRCcar(meta_data=params)
        elif self.planner_type == 'teleop':
            mpc_policy = TeleopMPCPolicyRCcar(meta_data=params)
        elif self.planner_type == 'lattice':
            additional_costs = []
            mpc_policy = LatticeMPCPolicyRCcar(self.trajopt,
                                               self.cost_cp,
                                               additional_costs=additional_costs,
                                               meta_data=params,
                                               use_threads=False,
                                               plot=True,
                                               epsilon_greedy=params['prediction']['dagger']['epsilon_greedy'])
        else:
            raise NotImplementedError('planner_type {0} not implemented for rccar'.format(self.planner_type))

        return mpc_policy

    ####################
    ### Info methods ###
    ####################

    def _get_world_info(self):
        ### just returns empty dict, but function call terminates bag recording
        return self.world.get_info()

