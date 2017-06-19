from general.policy.policy import Policy

class OpenLoopPolicy(Policy):

    def __init__(self, planner):
        self._planner = planner

    # TODO
    def act(self, obs_frame, t, rollout_num, only_noise=False, visualize=False):
        u, u_no_noise = self._planner.plan(obs_frame, t, rollout_num, only_noise=only_noise, visualize=visualize)
        return u, u_no_noise
