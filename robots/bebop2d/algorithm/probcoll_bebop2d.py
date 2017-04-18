import numpy as np
import openravepy as rave
import rospy
import os
from config import params
from general.algorithm.probcoll import Probcoll
from general.policy.open_loop_policy import OpenLoopPolicy
from general.state_info.conditions import Conditions
from general.state_info.sample import Sample

from robots.bebop2d.agent.agent_bebop2d import AgentBebop2d
from robots.bebop2d.algorithm.cost_probcoll_bebop2d import CostProbcollBebop2d
from robots.bebop2d.algorithm.probcoll_model_bebop2d import ProbcollModelBebop2d
from robots.bebop2d.dynamics.dynamics_bebop2d import DynamicsBebop2d
from robots.bebop2d.world.world_bebop2d import WorldBebop2d
from robots.bebop2d.planning.cost.cost_velocity_bebop2d import cost_velocity_bebop2d
from robots.bebop2d.planning.primitives_bebop2d import PrimitivesBebop2d

import std_msgs.msg as std_msgs
import general.ros.ros_utils as ros_utils
import IPython
class ProbcollBebop2d(Probcoll):

    def __init__(self, read_only=False):
        Probcoll.__init__(self, read_only=read_only)

    def _setup(self):
        # if hasattr(self, 'world'):
        #     self._world.destroy()
        # rave.RaveDestroy()
        rospy.init_node('DaggerPredictionBebop2d', anonymous=True)
        self.bad_rollout_callback = ros_utils.RosCallbackEmpty(params['bebop']['topics']['bad_rollout'], std_msgs.Empty)
        # pred_dagger_params = params['prediction']['dagger']
        probcoll_params = params['probcoll']
        world_params = params['world']
        # cond_params = pred_dagger_params['conditions']
        # cp_params = pred_dagger_params['cost_probcoll']
        # probcoll_params = params['probcoll']
        # self._max_iter = probcoll_params['max_iter']
        self._world = WorldBebop2d(self._bag_file, wp=world_params)
        # self._dynamics = DynamicsBebop2d()
        # self._agent = AgentBebop2d(self._world, self._dynamics,
        #                              obs_noise=probcoll_params['obs_noise'],
        #                              dyn_noise=probcoll_params['dyn_noise'])
        # # IPython.embed()
        # self._conditions = Conditions(cond_params=probcoll_params['conditions'])probcoll_params = params['probcoll']
        self._max_iter = probcoll_params['max_iter']
        # self._world = WorldBebop2d()
        self._dynamics = DynamicsBebop2d()
        self._agent = AgentBebop2d(self._world, self._dynamics,
                                     obs_noise=probcoll_params['obs_noise'],
                                     dyn_noise=probcoll_params['dyn_noise'])
        self._conditions = Conditions(cond_params=probcoll_params['conditions'])

        ### load prediction neural net
        self._probcoll_model = ProbcollModelBebop2d(read_only=self._read_only)
        self._cost_probcoll = CostProbcollBebop2d(self._probcoll_model)

    def _bag_file(self, itr, cond, rep, create=True):
        return os.path.join(self._itr_dir(itr, create=create), 'bagfile_itr{0}_cond{1}_rep{2}.bag'.format(itr, cond, rep))
    #####################
    ### World methods ###
    #####################

    def _reset_world(self, itr, cond, rep):
        self._agent.execute_control(None) # stop bebop
        self.bad_rollout_callback.get() # to clear it
        self._world.reset(itr, cond, rep)
        #
        # rave_bodies = [b for b in self._world.rave_env.env.GetBodies() if not b.IsRobot()]
        #
        # self._rave_cyl_bodies = []
        # self._rave_cyl_poses = []
        # self._rave_cyl_radii = []
        # self._rave_cyl_heights = []
        # self._rave_box_bodies = []
        # self._rave_box_poses = []
        # self._rave_box_extents = []
        # self._rave_other_bodies = []
        #
        # for body in rave_bodies:
        #     geom = body.GetLinks()[0].GetGeometries()[0]
        #     geom_type = geom.GetType()
        #
        #     pose = body.GetTransform()
        #
        #     if geom_type == rave.KinBody.Link.GeomType.Box:
        #         self._rave_box_bodies.append(body)
        #         self._rave_box_poses.append(pose)
        #         self._rave_box_extents.append(geom.GetBoxExtents())
        #     elif geom_type == rave.KinBody.Link.GeomType.Cylinder:
        #         self._rave_cyl_bodies.append(body)
        #         self._rave_cyl_poses.append(pose)
        #         self._rave_cyl_radii.append(geom.GetCylinderRadius())
        #         self._rave_cyl_heights.append(geom.GetCylinderHeight())
        #     else:
        #         self._rave_other_bodies.append(body)
        #
        # self._rave_cyl_poses = np.array(self._rave_cyl_poses)
        # self._rave_cyl_radii = np.array(self._rave_cyl_radii)
        # self._rave_cyl_heights = np.array(self._rave_cyl_heights)
        # self._rave_box_poses = np.array(self._rave_box_poses)
        # self._rave_box_extents = np.array(self._rave_box_extents)

    def _update_world(self, sample, t):
        return

        # laser_range = 1.5 * 5.0 # params['O']['laserscan']['range']
        # pos_robot = sample.get_X(t=t, sub_state='position')
        #
        # self._world.env.rave_env.update_local_environment(pos_robot, laser_range)

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

        self._logger.info('\t\tCreating MPC')

        cost_velocity = cost_velocity_bebop2d(self._probcoll_model.T,
                                                params['planning']['cost_velocity']['velocity'],
                                                params['planning']['cost_velocity']['weights'])

        if self._planner_type == 'primitives':
            planner = PrimitivesBebop2d(self._probcoll_model.T,
                                          self._dynamics,
                                          [cost_velocity, self._cost_probcoll],
                                          use_mpc=True)
            mpc_policy = OpenLoopPolicy(planner)
        else:
            raise Exception('Invalid planner type: {0}'.format(self._planner_type))

        return mpc_policy


    ####################
    ### Info methods ###
    ####################

    def _get_world_info(self):
        return self._world.get_info()
        # return {
        #         'cyl_poses': self._rave_cyl_poses,
        #         'cyl_radii': self._rave_cyl_radii,
        #         'cyl_heights': self._rave_cyl_heights,
        #         'box_poses': self._rave_box_poses,
        #         'box_extents': self._rave_box_extents
        #     }
