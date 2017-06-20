import numpy as np
from general.policy.policy import Policy
from config import params

class RandomPolicy(Policy):

    def act(self, obs_frame, t, rollout_num, only_noise=False, visualize=False):
        u = np.random.uniform(
            params['planning']['control_range']['lower'],
            params['planning']['control_range']['upper'])
        return u, u
