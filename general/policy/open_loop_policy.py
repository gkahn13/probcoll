from general.policy.policy import Policy

class OpenLoopPolicy(Policy):

    def __init__(self, planner):
        self._planner = planner

    def act(self, x, obs, t, noise):
        sample = self._planner.plan(x, obs)
        u = sample.get_U(t=0)
        return u + noise.sample(u)