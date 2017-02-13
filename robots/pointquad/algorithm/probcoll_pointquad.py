import numpy as np

import openravepy as rave

from general.algorithm.probcoll import Probcoll
from general.traj_opt.conditions import Conditions

from robots.pointquad.algorithm.probcoll_model_pointquad import ProbcollModelPointquad
from robots.pointquad.algorithm.cost_probcoll_pointquad import CostProbcollPointquad

from robots.pointquad.dynamics.dynamics_pointquad import DynamicsPointquad
from robots.pointquad.world.world_pointquad import WorldPointquad
from robots.pointquad.agent.agent_pointquad import AgentPointquad
from robots.pointquad.traj_opt.traj_opt_pointquad import TrajoptPointquad
from robots.pointquad.planning.primitives_pointquad import PrimitivesPointquad
from robots.pointquad.traj_opt.ilqr.cost.cost_velocity_pointquad import cost_velocity_pointquad

from general.state_info.sample import Sample
from general.policy.open_loop_policy import OpenLoopPolicy

from config import params

class ProbcollPointquad(Probcoll):

    def __init__(self, read_only=False):
        Probcoll.__init__(self, read_only=read_only)

    def _setup(self):
        # if hasattr(self, 'world'):
        #     self._world.destroy()
        # rave.RaveDestroy()
        probcoll_params = params['probcoll']

        self._max_iter = probcoll_params['max_iter']
        self._world = WorldPointquad()
        self._dynamics = DynamicsPointquad()
        self._agent = AgentPointquad(self._world, self._dynamics,
                                    obs_noise=probcoll_params['obs_noise'],
                                    dyn_noise=probcoll_params['dyn_noise'])
        self._trajopt = TrajoptPointquad(self._dynamics, self._world, self._agent)
        self._conditions = Conditions(cond_params=probcoll_params['conditions'])

        ### load prediction neural net
        self._probcoll_model = ProbcollModelPointquad(read_only=self._read_only)
        self._cost_probcoll = CostProbcollPointquad(self._probcoll_model)

    #####################
    ### World methods ###
    #####################

    def _reset_world(self, itr, cond, rep):
        self._world.reset(cond=cond)

        rave_bodies = [b for b in self._world.env.rave_env.env.GetBodies() if not b.IsRobot()]

        self._rave_cyl_bodies = []
        self._rave_cyl_poses = []
        self._rave_cyl_radii = []
        self._rave_cyl_heights = []
        self._rave_box_bodies = []
        self._rave_box_poses = []
        self._rave_box_extents = []
        self._rave_other_bodies = []

        for body in rave_bodies:
            geom = body.GetLinks()[0].GetGeometries()[0]
            geom_type = geom.GetType()

            pose = body.GetTransform()

            if geom_type == rave.KinBody.Link.GeomType.Box:
                self._rave_box_bodies.append(body)
                self._rave_box_poses.append(pose)
                self._rave_box_extents.append(geom.GetBoxExtents())
            elif geom_type == rave.KinBody.Link.GeomType.Cylinder:
                self._rave_cyl_bodies.append(body)
                self._rave_cyl_poses.append(pose)
                self._rave_cyl_radii.append(geom.GetCylinderRadius())
                self._rave_cyl_heights.append(geom.GetCylinderHeight())
            else:
                self._rave_other_bodies.append(body)

        self._rave_cyl_poses = np.array(self._rave_cyl_poses)
        self._rave_cyl_radii = np.array(self._rave_cyl_radii)
        self._rave_cyl_heights = np.array(self._rave_cyl_heights)
        self._rave_box_poses = np.array(self._rave_box_poses)
        self._rave_box_extents = np.array(self._rave_box_extents)

    def _update_world(self, sample, t):
        return

        # laser_range = 1.5 * 5.0 # params['O']['laserscan']['range']
        # pos_robot = sample.get_X(t=t, sub_state='position')
        #
        # self._world.env.rave_env.update_local_environment(pos_robot, laser_range)

    def _is_good_rollout(self, sample, t):
        return True

    #########################
    ### Create controller ###
    #########################

    def _create_mpc(self, itr, x0):
        """ Must initialize MPC """
        sample0 = Sample(meta_data=params, T=1)
        sample0.set_X(x0, t=0)
        self._update_world(sample0, 0)

        self._logger.info('\t\tCreating MPC')

        cost_velocity = cost_velocity_pointquad(self._probcoll_model.T,
                                                params['planning']['cost_velocity']['velocity'],
                                                params['planning']['cost_velocity']['weights'])

        if self._planner_type == 'primitives':
            planner = PrimitivesPointquad(self._probcoll_model.T,
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
        return {
                'cyl_poses': self.rave_cyl_poses,
                'cyl_radii': self.rave_cyl_radii,
                'cyl_heights': self.rave_cyl_heights,
                'box_poses': self.rave_box_poses,
                'box_extents': self.rave_box_extents
            }
