import numpy as np

from robots.pointquad.traj_opt.ilqr.cost.cost_velocity_pointquad import cost_velocity_pointquad

from rll_quadrotor.policy.primitives_mpc_policy import PrimitivesMPCPolicy

from rll_quadrotor.policy.policy import Policy
from rll_quadrotor.state_info.sample import Sample

class PrimitivePolicyPointquadAngleSpeed(Policy):

    def __init__(self, T, meta_data, theta, speed, dynamics):
        """
        :param T: time horizon
        :param meta_data
        :param theta: angle at which to move
        :param speed: m/s at which to move
        """
        Policy.__init__(self, T, False, meta_data)

        self.theta = theta
        self.speed = speed
        self.dynamics = dynamics

    def act(self, x, obs, t, noise, ref_traj=None):
        ### create desired path (radiates outward)
        T, dt, theta, speed = self._T, self._meta_data['dt'], self.theta, self.speed
        traj = Sample(meta_data=self._meta_data, T=T)

        ### initialize
        traj.set_X(x, t=0)
        traj.set_O(obs, t=0)

        ### create desired path (radiates outward)
        linearvel = [speed * np.cos(theta), speed * np.sin(theta), 0.]
        for t in xrange(T-1):
            x_t = traj.get_X(t=t)
            x_tp1 = self.dynamics.evolve(x_t, linearvel)

            traj.set_X(x_tp1, t=t+1)
            traj.set_U(linearvel, t=t)
        traj.set_U(linearvel, t=-1)

        self._curr_traj = traj

        u = traj.get_U(t=0)
        return u + noise.sample(u)

class PrimitivesMPCPolicyPointquad(PrimitivesMPCPolicy):

    def __init__(self, trajopt, cp_cost, meta_data=None,
                 obs_cost_func=None, additional_costs=[], replan_every=1,
                 plot=False, use_threads=False, epsilon_greedy=None):

        PrimitivesMPCPolicy.__init__(self, trajopt, cp_cost, meta_data=meta_data,
                                     obs_cost_func=obs_cost_func, additional_costs=additional_costs,
                                     replan_every=replan_every, plot=plot, use_threads=use_threads,
                                     epsilon_greedy=epsilon_greedy)

    ######################################
    ### Costs, primitives and policies ###
    ######################################

    def _create_additional_cost(self):
        """ Get to desired goal position """
        if 'cost_velocity' in self._meta_data['trajopt']:
            return cost_velocity_pointquad(self._T,
                                           self._meta_data['trajopt']['cost_velocity']['velocity'],
                                           self._meta_data['trajopt']['cost_velocity']['weights'],
                                           weight_scale=1.0)
        else:
            raise Exception('No additional cost function in yaml file')

    def _create_primitives_and_policies(self):
        primitives, mpc_policies = [], []

        if 'cost_velocity' in self._meta_data['trajopt']:
            des_vel = self._meta_data['trajopt']['cost_velocity']['velocity']
            weights = self._meta_data['trajopt']['cost_velocity']['weights']

            assert(weights[1] == 0 and weights[2] == 0)

            self.thetas = np.linspace(-np.pi/2., np.pi/2., 19) # 9
            self.speeds = np.linspace(0.1, 1.0, 10) * des_vel[0] # 10 # TODO

            for theta in self.thetas:
                for speed in self.speeds:

                    primitives.append(None)
                    mpc_policies.append(PrimitivePolicyPointquadAngleSpeed(
                        self._T, self._meta_data, theta, speed, self._trajopt.dynamics))

        return primitives, mpc_policies

    def _create_primitive_cost_func(self, x, primitive):
        """ Follow primitive path from current position """
        return None # not using LQR

    ####################
    ### Data methods ###
    ####################

    def _plot(self, costs, samples):
        pass # TODO
