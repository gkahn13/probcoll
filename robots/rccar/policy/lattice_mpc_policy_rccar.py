import itertools
import numpy as np

import rospy
import visualization_msgs.msg as vm
import geometry_msgs.msg as gm

from robots.rccar.traj_opt.ilqr.cost.cost_velocity_rccar import cost_velocity_rccar

from rll_quadrotor.policy.primitives_mpc_policy import PrimitivesMPCPolicy

from rll_quadrotor.policy.policy import Policy
from rll_quadrotor.state_info.sample import Sample

class LatticePolicyRCcar(Policy):
    def __init__(self, T, meta_data, U, dynamics):
        Policy.__init__(self, T, False, meta_data)

        self.U = U
        self.dynamics = dynamics

    def act(self, x, obs, t, noise, ref_traj=None):
        ### create desired path (radiates outward)
        traj = Sample(meta_data=self._meta_data, T=self._T)

        ### initialize
        traj.set_X(x, t=0)
        traj.set_O(obs, t=0)
        traj.set_U(self.U, t=slice(0, self._T))
        traj.rollout(self.dynamics)

        self._curr_traj = traj

        u = traj.get_U(t=0)
        return u + noise.sample(u)

class LatticeMPCPolicyRCcar(PrimitivesMPCPolicy):

    def __init__(self, trajopt, cp_cost, meta_data=None,
                 obs_cost_func=None, additional_costs=[], replan_every=1,
                 plot=False, use_threads=False, epsilon_greedy=None):

        PrimitivesMPCPolicy.__init__(self, trajopt, cp_cost, meta_data=meta_data,
                                     obs_cost_func=obs_cost_func, additional_costs=additional_costs,
                                     replan_every=replan_every, plot=plot, use_threads=use_threads,
                                     epsilon_greedy=epsilon_greedy)

        rccar_topics = meta_data['rccar']['topics']
        self.debug_cost_pub = rospy.Publisher(rccar_topics['debug_cost'],
                                              vm.MarkerArray,
                                              queue_size=10)

    ######################################
    ### Costs, primitives and policies ###
    ######################################

    def _create_additional_cost(self):
        """ Get to desired goal position """
        if 'cost_velocity' in self._meta_data['trajopt']:
            return cost_velocity_rccar(self._T,
                                       self._meta_data['trajopt']['cost_velocity']['u_des'],
                                       self._meta_data['trajopt']['cost_velocity']['u_weights'],
                                       weight_scale=1.0)
        else:
            raise Exception('No additional cost function in yaml file')

    def _create_primitives_and_policies(self):
        primitives, mpc_policies = [], []

        lattice_params = self._meta_data['mpc']['lattice']
        u_min = lattice_params['min']
        u_max = lattice_params['max']
        branching = lattice_params['branching']
        assert(branching[1] == 0) # no branching on speed

        u_lattice = [(steer, u_min[1]) for steer in
                    # [10., 20., 30., 40., 42., 44., 46., 48., 50., 52., 54., 56., 58., 60., 70., 80., 90.]]
                     np.linspace(u_min[0], u_max[0], branching[0])]

        for U in itertools.product(*[u_lattice] * self._T):
            primitives.append(None)
            mpc_policies.append(LatticePolicyRCcar(self._T, self._meta_data, U, self._trajopt.dynamics))

        return primitives, mpc_policies

    def _create_primitive_cost_func(self, x, primitive):
        """ Follow primitive path from current position """
        return None # not using LQR

    ####################
    ### Data methods ###
    ####################

    def _plot(self, costs, samples):
        marker_array = vm.MarkerArray()

        sample = samples[np.argmin(costs)]

        origin = np.array([0., 0., 0.])
        angle = 0.
        for t in xrange(len(sample.get_U())):
            marker = vm.Marker()
            marker.id = 0 * len(sample.get_U()) + t
            marker.header.frame_id = '/map'
            marker.type = marker.ARROW
            marker.action = marker.ADD

            speed = sample.get_U(t=t, sub_control='cmd_vel')[0]
            steer = sample.get_U(t=t, sub_control='cmd_steer')[0]
            angle += (steer - 50.) / 100. * (np.pi / 2)
            new_origin = origin + [speed * np.cos(angle), speed * np.sin(angle), 0.]
            marker.points = [
                gm.Point(*origin.tolist()),
                gm.Point(*new_origin.tolist())
            ]
            origin = new_origin

            marker.lifetime = rospy.Duration()
            marker.scale.x = 0.05
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            marker.color.r, marker.color.g, marker.color.b, marker.color.a = (0., 0., 1., 1.)

            marker_array.markers.append(marker)

        self.debug_cost_pub.publish(marker_array)
