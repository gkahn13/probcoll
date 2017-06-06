from general.policy.policy import Policy

class OpenLoopPolicy(Policy):

    def __init__(self, planner):
        self._planner = planner

    # TODO
    def act(self, x, obs, t):
        u, u_no_noise = self._planner.plan(x, obs, t)
        return u, u_no_noise
