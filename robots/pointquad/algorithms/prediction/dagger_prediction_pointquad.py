import pickle, itertools
import numpy as np

import openravepy as rave

from general.algorithms.prediction.dagger_prediction import DaggerPrediction
from general.algorithms.prediction.cost_prediction import CostPredictionGroundTruth
from general.traj_opt.conditions import Conditions

from robots.pointquad.algorithms.prediction.prediction_model_pointquad import PredictionModelPointquad
from robots.pointquad.algorithms.prediction.cost_prediction_pointquad import CostPredictionPointquad

from robots.pointquad.dynamics.dynamics_pointquad import DynamicsPointquad
from robots.pointquad.world.world_pointquad import WorldPointquad
from robots.pointquad.agent.agent_pointquad import AgentPointquad
from robots.pointquad.traj_opt.traj_opt_pointquad import TrajoptPointquad
from robots.pointquad.policy.primitives_mpc_policy_pointquad import PrimitivesMPCPolicyPointquad
from robots.pointquad.traj_opt.ilqr.cost.cost_velocity_pointquad import cost_velocity_pointquad

from rll_quadrotor.policy.random_walk_policy import RandomWalkPolicy
from rll_quadrotor.policy.random_control_set_policy import RandomControlSetPolicy
from rll_quadrotor.policy.mpc_policy import MPCPolicy
from rll_quadrotor.policy.cem_mpc_policy import CEMMPCPolicy
from rll_quadrotor.state_info.sample import Sample
from rll_quadrotor.utility.utils import to_hist

from config import params

class DaggerPredictionPointquad(DaggerPrediction):

    def __init__(self, read_only=False):
        DaggerPrediction.__init__(self, read_only=read_only)

    def _setup(self):
        # if hasattr(self, 'world'):
        #     self.world.destroy()
        # rave.RaveDestroy()
        pred_dagger_params = params['prediction']['dagger']
        world_params = params['world']
        cond_params = pred_dagger_params['conditions']
        cp_params = pred_dagger_params['cost_prediction']

        self.max_iter = pred_dagger_params['max_iter']
        self.world = WorldPointquad(wp=world_params, view_rave=world_params['view_rave'])
        self.dynamics = DynamicsPointquad()
        self.agent = AgentPointquad(self.world, self.dynamics,
                                    obs_noise=pred_dagger_params['obs_noise'],
                                    dyn_noise=pred_dagger_params['dyn_noise'])
        self.trajopt = TrajoptPointquad(self.dynamics, self.world, self.agent)
        self.conditions = Conditions(cond_params=cond_params)

        ### load prediction neural net (must go after self.world creation, why??)
        self.bootstrap = PredictionModelPointquad(read_only=self.read_only)

        if pred_dagger_params['use_init_cost']:
            self.cost_cp_init = cost_velocity_pointquad(params['mpc']['H'],
                                                        0.4 * np.array(params['trajopt']['cost_velocity']['velocity']),
                                                        [1., 1., 1.],
                                                        weight_scale=1e15)
        else:
            self.cost_cp_init = None

        if not pred_dagger_params['use_ground_truth']:
            self.cost_cp = CostPredictionPointquad(self.bootstrap, self.agent,
                                                   weight=float(cp_params['weight']),
                                                   eval_cost=cp_params['eval_cost'],
                                                   pre_activation=cp_params['pre_activation'])
        else:
            self.cost_cp = CostPredictionGroundTruth(self.bootstrap,
                                                     world=self.world,
                                                     weight=float(cp_params['weight']),
                                                     eval_cost=cp_params['eval_cost'],
                                                     pre_activation=cp_params['pre_activation'])

    #####################
    ### World methods ###
    #####################

    def _reset_world(self, itr, cond, rep):
        self.world.reset(cond=cond)

        rave_bodies = [b for b in self.world.env.rave_env.env.GetBodies() if not b.IsRobot()]

        self.rave_cyl_bodies = []
        self.rave_cyl_poses = []
        self.rave_cyl_radii = []
        self.rave_cyl_heights = []
        self.rave_box_bodies = []
        self.rave_box_poses = []
        self.rave_box_extents = []
        self.rave_other_bodies = []

        for body in rave_bodies:
            geom = body.GetLinks()[0].GetGeometries()[0]
            geom_type = geom.GetType()

            pose = body.GetTransform()

            if geom_type == rave.KinBody.Link.GeomType.Box:
                self.rave_box_bodies.append(body)
                self.rave_box_poses.append(pose)
                self.rave_box_extents.append(geom.GetBoxExtents())
            elif geom_type == rave.KinBody.Link.GeomType.Cylinder:
                self.rave_cyl_bodies.append(body)
                self.rave_cyl_poses.append(pose)
                self.rave_cyl_radii.append(geom.GetCylinderRadius())
                self.rave_cyl_heights.append(geom.GetCylinderHeight())
            else:
                self.rave_other_bodies.append(body)

        self.rave_cyl_poses = np.array(self.rave_cyl_poses)
        self.rave_cyl_radii = np.array(self.rave_cyl_radii)
        self.rave_cyl_heights = np.array(self.rave_cyl_heights)
        self.rave_box_poses = np.array(self.rave_box_poses)
        self.rave_box_extents = np.array(self.rave_box_extents)

    def _update_world(self, sample, t):
        return

        # laser_range = 1.5 * 5.0 # params['O']['laserscan']['range']
        # pos_robot = sample.get_X(t=t, sub_state='position')
        #
        # self.world.env.rave_env.update_local_environment(pos_robot, laser_range)

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

        additional_costs = []
        # if self.use_cp_cost:
        #     # additional_costs += [self.cost_cp, self.cost_cage] # TODO
        #     additional_costs += [self.cost_cp]
        #     # additional_costs += [self.cost_cage]

        self.logger.info('\t\tCreating MPC')

        if self.planner_type == 'primitives':
            if itr <= 1 and self.cost_cp_init is not None:
                additional_costs.append(self.cost_cp_init)
            mpc_policy = PrimitivesMPCPolicyPointquad(self.trajopt,
                                                      self.cost_cp,
                                                      additional_costs=additional_costs, # TODO
                                                      meta_data=params,
                                                      use_threads=False,
                                                      plot=True,
                                                      epsilon_greedy=params['prediction']['dagger']['epsilon_greedy'])
        elif self.planner_type == 'cem':
            if itr <= 1 and self.cost_cp_init is not None:
                costs = [self.cost_cp_init]
            else:
                if 'cost_velocity' in params['trajopt']:
                    additional_costs.append(cost_velocity_pointquad(params['mpc']['H'],
                                                                    params['trajopt']['cost_velocity']['velocity'],
                                                                    params['trajopt']['cost_velocity']['weights'],
                                                                    weight_scale=1.0))
                else:
                    raise Exception('No additional cost function in yaml file')
                costs = [self.cost_cp] + additional_costs
            mpc_policy = CEMMPCPolicy(self.world.env,
                                      self.dynamics,
                                      costs,
                                      meta_data=params)
        elif self.planner_type == 'ilqr':
            assert(False) # ilqr not ready yet
            mpc_policy = MPCPolicy(self.trajopt,
                                   meta_data=params,
                                   additional_costs=additional_costs,
                                   plot=True)
        elif self.planner_type == 'randomwalk':
            des_vel = params['trajopt']['cost_velocity']['velocity'][0]
            lb = [0., -1 * des_vel, 0.],
            ub = [1.5 * des_vel, 1 * des_vel, 0.]
            init_u = np.random.uniform([0., -0.2 * des_vel, 0.], [1.5 * des_vel, 0.2 * des_vel, 0.])
            mpc_policy = RandomWalkPolicy(meta_data=params,
                                          init_u=init_u,
                                          lower_bound=lb,
                                          upper_bound=ub)
        elif self.planner_type == 'randomcontrolset':
            prim_policy = PrimitivesMPCPolicyPointquad(self.trajopt,
                                                       self.cost_cp,
                                                       additional_costs=additional_costs,
                                                       meta_data=params,
                                                       use_threads=False,
                                                       plot=True,
                                                       epsilon_greedy=params['prediction']['dagger']['epsilon_greedy'])
            control_set = [(speed * np.cos(theta), speed * np.sin(theta), 0)
                           for speed in prim_policy.speeds for theta in prim_policy.thetas]
            mpc_policy = RandomControlSetPolicy(meta_data=params,
                                                control_set=control_set)

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
