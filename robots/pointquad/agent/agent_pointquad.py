import numpy as np

from general.agent.agent import Agent

from rll_quadrotor.simulation.simulator import Simulator

from config import params

class AgentPointquad(Agent):
    def __init__(self, world, dynamics, obs_noise=False, dyn_noise=False):
        self.world = world
        self.dynamics = dynamics

        self.simulator = Simulator(self.world.env, self.dynamics, meta_data=params,
                                   obs_noise=obs_noise, dyn_noise=dyn_noise)

    def sample_policy(self, x0, policy, T=None, **policy_args):
        """
        Run the policy and collect the trajectory data

        :param x0: initial state
        :param policy: to execute
        :param policy_args: e.g. ref_traj, noise, etc
        :rtype Sample
        """
        return self.simulator.sample_policy(x0, policy, T=T, **policy_args)

    def reset(self, x):
        """
        Reset the simulated environment as the specified state.
        Return the actual model state vector.

        :param x: state vector to reset to
        :rtype: np.ndarray
        """
        return np.copy(x)

    def get_observation(self, x, noise=True):
        return self.simulator.get_observation(np.copy(x), np.copy(x), noise=noise)
