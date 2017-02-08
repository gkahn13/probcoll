from general.state_info.sample import Sample
from general.policy.policy import Policy
from general.utility.logger import get_logger

class StraightPolicyRCcar(Policy):

    def __init__(self, meta_data):
        use_obs = False
        Policy.__init__(self, meta_data['mpc']['H'], use_obs, meta_data)
        self.logger = get_logger(self.__class__.__name__, 'warn')

    def act(self, x, obs, t, noise, ref_traj=None):
        s = Sample(meta_data=self._meta_data, T=2)
        s.set_U([50.], t=0, sub_control='cmd_steer')
        s.set_U([2.5], t=0, sub_control='cmd_vel')
        u = s.get_U(t=0)

        return u + noise.sample(u)

    def get_info(self):
        return dict()
