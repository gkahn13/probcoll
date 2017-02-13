import numpy as np

import rospy
import visualization_msgs.msg as vm
import geometry_msgs.msg as gm

from robots.rccar.traj_opt.ilqr.cost.cost_velocity_rccar import cost_velocity_rccar

from rll_quadrotor.policy.primitives_mpc_policy import PrimitivesMPCPolicy

from general.policy.policy import Policy
from general.state_info.sample import Sample

class PrimitivePolicyRCcarSteerVelocity(Policy):
    def __init__(self, T, meta_data, steers, vels, dynamics):
        assert(len(steers) == T)
        assert(len(vels) == T)
        Policy.__init__(self, T, False, meta_data)

        self.steers = steers
        self.vels = vels
        self._dynamics = dynamics

    def act(self, x, obs, t, noise, ref_traj=None):
        ### create desired path (radiates outward)
        T, dt = self._T, self._meta_data['dt']
        traj = Sample(meta_data=self._meta_data, T=T)

        ### initialize
        traj.set_X(x, t=0)
        traj.set_O(obs, t=0)
        traj.set_U(np.expand_dims(self.steers, 1), t=slice(0, T), sub_control='cmd_steer')
        traj.set_U(np.expand_dims(self.vels, 1), t=slice(0, T), sub_control='cmd_vel')

        ### create desired path (radiates outward)
        for t in xrange(T-1):
            u_t = traj.get_U(t=t)
            x_t = traj.get_X(t=t)
            x_tp1 = self._dynamics.evolve(x_t, u_t)

            traj.set_X(x_tp1, t=t+1)

        self._curr_traj = traj

        u = traj.get_U(t=0)
        return u + noise.sample(u)

class PrimitivesMPCPolicyRCcar(PrimitivesMPCPolicy):

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

        if 'cost_velocity' in self._meta_data['trajopt']:
            des_steer, des_vel = self._meta_data['trajopt']['cost_velocity']['u_des'] # TODO: hard-coded

            # TODO: may need to be more clever
            # constant steer
            # des_steers_list = [[s] * self._T for s in np.linspace(10., 90., 18)]
            # first steer one direction, then the other
            des_steers_list = []
            for t in xrange(0, self._T):
                # for steer in np.linspace(10., 90., 11):
                for steer in [10., 30., 45., 50., 55., 70., 90.]:
                    des_steers_list.append([steer]*t + [100 - steer]*(self._T - t))
            # constant speed
            # des_vels_list = [[v] * self._T for v in np.linspace(0.1, 1, 8) * des_vel]
            assert(abs(des_vel - 14) < 1e-4)
            min_vel = self._meta_data['U']['cmd_vel']['min']
            assert (abs(min_vel - 6) < 1e-4)
            des_vels_list = [[speed] * self._T for speed in min_vel + np.array([2, 4, 6, 8])] # TODO: temp, constant speed
            # des_vels_list = [[1.5] * self._T]

            for des_steers in des_steers_list:
                for des_vels in des_vels_list:

                    primitives.append(None)
                    mpc_policies.append(PrimitivePolicyRCcarSteerVelocity(
                        self._T, self._meta_data, des_steers, des_vels, self._trajopt.dynamics))

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

        marker = vm.Marker()
        marker.id = 0
        marker.header.frame_id = '/map'
        marker.type = marker.TEXT_VIEW_FACING
        marker.action = marker.ADD
        marker.text = '\n'.join(['({0:.1f}, {1:.2f})'.format(u[0], u[1]) for u in sample.get_U()])
        marker.pose.position.x = -4.
        marker.pose.position.y = -3.
        marker.scale.z = 1.
        marker.color.r = 1.
        marker.color.g = 1.
        marker.color.b = 1.
        marker.color.a = 1.
        marker.pose.orientation.w = 1.
        marker_array.markers.append(marker)

        origin = np.array([0., 0., 0.5])
        angle = 0.
        for t in xrange(len(sample.get_U())):
            marker = vm.Marker()
            marker.id = 1 + 0 * len(sample.get_U()) + t
            marker.header.frame_id = '/map'
            marker.type = marker.ARROW
            marker.action = marker.ADD

            speed = sample.get_U(t=t, sub_control='cmd_vel')[0] / 5.
            steer = sample.get_U(t=t, sub_control='cmd_steer')[0]
            angle += (steer - 50.) / 100. * (np.pi / 2)
            new_origin = origin + [speed * np.cos(angle), speed * np.sin(angle), 0.]
            marker.points = [
                gm.Point(*origin.tolist()),
                gm.Point(*new_origin.tolist())
            ]
            origin = new_origin

            marker.lifetime = rospy.Duration()
            marker.scale.x = 0.05*4
            marker.scale.y = 0.1*4
            marker.scale.z = 0.1*4

            marker.color.r, marker.color.g, marker.color.b, marker.color.a = (0., 0., 1., 1.)

            marker_array.markers.append(marker)

        self.debug_cost_pub.publish(marker_array)
