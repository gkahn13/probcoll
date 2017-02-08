import numpy as np

from general.policy.policy import Policy
from general.planning.cem.cem import CEM
from general.state_info.sample import Sample

from config import params as default_meta_data

class CEMMPCPolicy(Policy):

    def __init__(self, env, dynamics, cost_funcs, meta_data=None):
        use_obs = False  # MPC must not accept obs input
        meta_data = meta_data if meta_data is not None else default_meta_data
        Policy.__init__(self, meta_data['mpc']['H'], use_obs, meta_data)

        self.cem = CEM(meta_data=meta_data, config=meta_data['mpc']['cem']['init'])

        self.env = env
        self.dynamics = dynamics
        self.cost_funcs = cost_funcs

        self.init_traj = None
        self.controls = []
        self.means, self.covs = [], []
        self._curr_traj = None

    def act(self, x, obs, t, noise):
        if len(self.controls) > 0 and self._meta_data['mpc']['cem']['warm_start']:
            prev_controls = self.controls[-1]
            config = self._meta_data['mpc']['cem']['warm_start_config']
            init_cov = self.cem.covs[-1]
            ### update init_traj
            self.init_traj = Sample(meta_data=self._meta_data, T=self._T)
            U = np.vstack((prev_controls[1:], prev_controls[-1]))
            self.init_traj.set_U(U, t=slice(0, self._T))
            self.init_traj.set_X(x, t=0)
            self.init_traj.rollout(self.dynamics)
            self.init_traj.set_O(obs, t=0)
        else:
            config = self._meta_data['mpc']['cem']['init']
            init_cov = None

        ### plan using CEM
        self._curr_traj = self.cem.plan(x, obs, self.dynamics, self.cost_funcs, self._udim, self._T,
                                        init_traj=self.init_traj, plot_env=None, init_cov=init_cov, config=config)

        ### record
        controls = self._curr_traj.get_U()
        self.controls.append(controls)
        self.means.append(self.cem.means)
        self.covs.append(self.cem.covs)

        u = controls[0]
        return u + noise.sample(u)


    def get_info(self):
        """
        :param file_path: .pkl
        :return info dict
        """
        d = {
            'controls': self.controls,
            'means': self.means,
            'covs': self.covs
        }

        ### reset
        self.controls = []
        self.means = []
        self.covs = []

        return d
