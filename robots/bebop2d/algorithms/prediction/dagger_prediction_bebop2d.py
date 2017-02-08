import os

import rospy
import std_msgs.msg as std_msgs

from general.algorithm.prediction.probcoll import Probcoll
from general.traj_opt.conditions import Conditions
import general.ros.ros_utils as ros_utils

from robots.bebop2d.algorithm.prediction.probcoll_model_bebop2d import ProbcollModelBebop2d
from robots.bebop2d.algorithm.prediction.cost_probcoll_bebop2d import CostProbcollBebop2d

from robots.bebop2d.dynamics.dynamics_bebop2d import DynamicsBebop2d
from robots.bebop2d.world.world_bebop2d import WorldBebop2d
from robots.bebop2d.agent.agent_bebop2d import AgentBebop2d
from robots.bebop2d.traj_opt.traj_opt_bebop2d import TrajoptBebop2d
from robots.bebop2d.policy.primitives_mpc_policy_bebop2d import PrimitivesMPCPolicyBebop2d
from robots.bebop2d.policy.teleop_mpc_policy_bebop2d import TeleopMPCPolicyBebop2d

from general.state_info.sample import Sample

from config import params

class ProbcollBebop2d(Probcoll):

    def __init__(self, read_only=False):
        Probcoll.__init__(self, read_only=read_only)

    def _setup(self):
        rospy.init_node('ProbcollBebop2d', anonymous=True)

        self.bad_rollout_callback = ros_utils.RosCallbackEmpty(params['bebop']['topics']['bad_rollout'], std_msgs.Empty)

        pred_dagger_params = params['prediction']['dagger']
        world_params = params['world']
        cond_params = pred_dagger_params['conditions']
        cp_params = pred_dagger_params['cost_probcoll']

        self.max_iter = pred_dagger_params['max_iter']
        self.world = WorldBebop2d(self._bag_file, wp=world_params)
        self.dynamics = DynamicsBebop2d()
        self.agent = AgentBebop2d(self.dynamics)
        self.trajopt = TrajoptBebop2d(self.dynamics, self.world, self.agent)
        self.conditions = Conditions(cond_params=cond_params)

        assert(self.world.randomize)

        ### load prediction neural net (must go after self.world creation, why??)
        self.bootstrap = ProbcollModelBebop2d(read_only=self.read_only)

        self.cost_cp = CostProbcollBebop2d(self.bootstrap,
                                             weight=float(cp_params['weight']),
                                             eval_cost=cp_params['eval_cost'],
                                             pre_activation=cp_params['pre_activation'])

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
        self.agent.execute_control(None) # stop bebop
        self.bad_rollout_callback.get() # to clear it
        self.world.reset(itr=itr, cond=cond, rep=rep)

    def _update_world(self, sample, t):
        return

    def _is_good_rollout(self, sample, t):
        return self.bad_rollout_callback.get() is None

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
            mpc_policy = PrimitivesMPCPolicyBebop2d(self.trajopt,
                                                    self.cost_cp,
                                                    additional_costs=additional_costs,
                                                    meta_data=params,
                                                    use_threads=False,
                                                    plot=True,
                                                    epsilon_greedy=params['prediction']['dagger']['epsilon_greedy'])
        elif self.planner_type == 'teleop':
            mpc_policy = TeleopMPCPolicyBebop2d(params)
        else:
            raise NotImplementedError('planner_type {0} not implemented for bebop2d'.format(self.planner_type))

        return mpc_policy

    ####################
    ### Info methods ###
    ####################

    def _get_world_info(self):
        ### just returns empty dict, but function call terminates bag recording
        return self.world.get_info()

