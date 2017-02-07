import threading
import numpy as np
from general.policy.policy import LinearPolicy

from general.utility.utils import finite_differences

from rll_quadrotor.planning.ilqr.cost.cost import Cost
from rll_quadrotor.planning.ilqr.util.approx import CostApprox
from rll_quadrotor.utility.logger import get_logger


class CostKL(Cost):
    """
    Computes the additional term for augmented cost function for KL constraint
    using linear approximation.
    i.e. - \rho * \log \pi(u_t | x_t)
    \pi(u | x) \approx \pi(x_ref) + (dmu / dx around x_ref) (x - x_ref)
    """

    def __init__(self, **kwargs):
        Cost.__init__(self)
        self._logger = get_logger(self.__class__.__name__, 'debug')

        self._rho = kwargs.pop('rho')
        self._policy = kwargs.pop('policy', None)
        self._k = kwargs.pop('k', None)
        self._K = kwargs.pop('K', None)
        self._T = kwargs.pop('T', None)
        self._use_obs = kwargs.pop('use_obs', False)
        self._agent = kwargs.pop('agent', None)
        self._lock = threading.RLock()

    def eval(self, sample):
        with self._lock:
            T = sample._T if self._T is None else self._T
            dX = sample._xdim
            dU = sample._udim
            if self._k is not None and self._K is not None:
                k, K = list(np.array(self._k)), list(np.array(self._K))
            elif isinstance(self._policy, LinearPolicy):
                # TODO: only work with caffe
                # k, K = self._policy.linearize_all(sample)
                k, K = self._linearize_caffe(sample)
            else:
                raise RuntimeError("cannot linearize policy")

            for t in xrange(T):
                k[t] -= sample.get_U(t=t)

            chol_invSigma = self._policy.chol_pol_covar
            invSigma = chol_invSigma.dot(chol_invSigma.T)
            cst_approx = CostApprox(sample._T, dX, dU) # must be same T as sample
            for t in range(T):
                cst_approx.set_t(
                    t,
                    l=0.5 * k[t].T.dot(invSigma.dot(k[t])),
                    lu=-self._rho * invSigma.dot(k[t]),
                    lx=self._rho * K[t].T.dot(invSigma.dot(k[t])) if not self._use_obs else 0,
                    lxx=self._rho * K[t].T.dot(invSigma.dot(K[t])) if not self._use_obs else 0,
                    luu=self._rho * invSigma,
                    lux=-self._rho * invSigma.dot(K[t]) if not self._use_obs else 0
                )

        return cst_approx

    def _linearize_caffe(self, sample):
        T = sample._T if self._T is None else self._T
        dX = sample._xdim
        dU = sample._udim

        if not self._use_obs:
            k, K = self._policy.linearize_all(sample.match(slice(0,T)))
        else:
            assert(self._policy is not None and self._agent is not None)

            ##############
            # Only set k #
            ##############

            def obs_func(x):
                s = self._agent.sample_policy(x, self._policy, noise=False, T=1)
                return s.get_O(t=0)

            k, K = [], []
            for t, x in enumerate(sample.get_X()):
                o = obs_func(x)
                u = self._policy.act(x, o, t, noise=False)
                k.append(u)
                K.append(np.zeros((dX, dU)))


            #############################
            # Finite differences method #
            #############################
            # def obs_func(x):
            #     s = self.agent.sample_policy(x, self._policy, noise=False, T=1)
            #     return s.get_O(t=0)
            #
            # ### fill in sample with observations
            # for t, x in enumerate(sample.get_X()):
            #     sample.set_O(obs_func(x), t=t)
            # ### calculate dU/dO via backprop
            # k, dUdO = self._policy.linearize_all(sample)
            # ### calculate dO/dX via finite differences
            # dOdX = [finite_differences(x, obs_func) for x in sample.get_X()]
            # ### dU/dX = dU/dO * dO/dX
            # K = [A.dot(B) for A, B in zip(dUdO, dOdX)]

        return k, K

    def compare(self, sample):
        # TODO: not sure if works
        if self._k is None:
            return None, None

        T = sample._time_meta.T
        invSigma = self._policy.invSigma

        kl_real = []
        kl_approx = []
        for t in xrange(T):
            try:
                u_pi = self._policy.act(sample.get_X(t=t),
                                        sample.get_O(t=t), t)[1]
                u_pi_k = self._k[t]
                u_t = sample.get_U(t=t)
                kl_real.append((u_t - u_pi).T.dot(invSigma).dot(u_t - u_pi))
                kl_approx.append((u_t - u_pi_k).T.dot(invSigma).dot(u_t - u_pi_k))
                self._logger.debug('{0}: kl_real, kl_approx: {1}\t{2}'.format(
                    t, kl_real[-1], kl_approx[-1]
                ))
            except:
                pass

        kl_real_sum = np.sum([kl for kl in kl_real if np.isfinite(kl)])
        kl_approx_sum = np.sum([kl for kl in kl_approx if np.isfinite(kl)])

        return kl_real_sum, kl_approx_sum

        # def reject_outliers(data, m):
        #     return data[abs(data - np.mean(data)) < m * np.std(data)]
        #
        # diff_pcts = abs((kl_real - kl_approx) / kl_real)
        # diff_pcts_filt = reject_outliers(diff_pcts, m=1.)
        #
        # return np.mean(diff_pcts_filt), np.std(diff_pcts_filt), np.max(diff_pcts)