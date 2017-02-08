import numpy as np

from .policy import Policy, LinearPolicy

from rll_quadrotor.policy.noise_models import ZeroNoise, GaussianNoise
from rll_quadrotor.config import meta_data

class LinearGaussianPolicy(Policy, LinearPolicy):
    """
    Time-varying linear gaussian policy to follow a reference a trajectory.
    u = u_ref + k + K * (x - x_ref) + noise s.t. noise ~ N(0, chol_pol_covar)
    If reference is not provided, then u = k + K * x + noise
    """

    def __init__(self, T, **kwargs):
        """
        Accepted kwargs are:
            k:
            K:
            pol_covar:
            chol_pol_covar:
            inv_pol_covar:
            init:
        """
        use_obs = False  # LQR cannot accept obs
        self._meta_data = kwargs.pop('meta_data', meta_data)
        Policy.__init__(self, T, use_obs, self._meta_data)
        self.pol_covar = kwargs.pop('pol_covar',
            np.zeros((self._T, self._udim, self._udim)))
        self.inv_pol_covar = kwargs.pop('inv_pol_covar',
            np.zeros((self._T, self._udim, self._udim)))
        self.chol_pol_covar = kwargs.pop('chol_pol_covar',
            np.zeros((self._T, self._udim, self._udim)))
        self.k = kwargs.pop('k',
            np.nan * np.ones((self._T, self._udim)))
        self.K = kwargs.pop('K',
            np.nan * np.ones((self._T, self._udim, self._xdim)))

        if kwargs.pop('init', False):
            self.K[:] = 0.
            self.k[:] = 0.

    def act(self, x, obs, t, noise, ref_traj=None):
        # assert obs is None, 'LinearGaussianPolicy does not take in obs'
        assert(ref_traj is not None)
        assert((type(noise) == ZeroNoise) or ((type(noise) == GaussianNoise) and np.allclose(noise.std, noise.std[0])))
        x_ref = ref_traj.get_X(t=t)
        u_ref = ref_traj.get_U(t=t)
        u = u_ref + self.k[t] + self.K[t].dot(x - x_ref)
        u += self.chol_pol_covar[t].T.dot(noise.sample(u))
        return u

    def linearize(self, x=None, u=None, t=None):
        assert t is not None
        return self.k[t], self.K[t]

    def linearize_all(self, traj):
        return np.copy(self.k), np.copy(self.K)

    def empty_like(self):
        return self.__class__(self._T, meta_data=self._meta_data)

    def match(self, time_slice):
        """
        :type time_slice: slice
        :return: LinearGaussianPolicy
        """
        T = time_slice.stop - time_slice.start

        lgp = LinearGaussianPolicy(T, meta_data=self._meta_data,
                                   pol_covar=self.pol_covar[time_slice],
                                   inv_pol_covar=self.inv_pol_covar[time_slice],
                                   chol_pol_covar=self.chol_pol_covar[time_slice],
                                   k=self.k[time_slice],
                                   K=self.K[time_slice])
        return lgp
