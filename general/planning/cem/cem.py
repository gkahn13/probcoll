import numpy as np

from general.state_info.sample import Sample
from general.utility.logger import get_logger

from config import params as default_meta_data

class CEM(object):
    def __init__(self, meta_data=None, config=None):
        self._meta_data = meta_data if meta_data is not None else default_meta_data
        self._config = config if config is not None else self._meta_data['cem']
        self._logger = get_logger(self.__class__.__name__, 'fatal')

        self.means, self.covs = [], [] # for recording

    def plan(self, x0, obs, dynamics, cost_funcs, dU, T, init_traj=None, plot_env=None, init_cov=None, config=None):
        """
        :type init_traj: Sample
        :type dynamics: Dynamics
        :type cost_func: Cost
        """
        if config is None:
            config = self._config
        self.means, self.covs = [], []

        if init_traj is not None:
            mean = init_traj.get_U().ravel()
        elif 'init_u' in config:
            mean = config['init_u'] * T
        else:
            mean = np.zeros(dU * T)

        lower_bound = config['bounds']['min']*T if 'bounds' in config else -np.inf * np.ones(dU*T)
        upper_bound = config['bounds']['max']*T if 'bounds' in config else np.inf * np.ones(dU*T)

        ### run one iteration of CEM with extra samples
        # if init_cov is None:
        cov = config['init_var'] * np.eye(dU*T)
        # else:
        #     cov = config['init_var'] * init_cov
        mean, cov = self._cem_step(x0, obs, mean, cov, config['init_M'], config['K'], lower_bound, upper_bound,
                                   dU, T, dynamics, cost_funcs, plot_env=plot_env)
        self.means.append(mean)
        self.covs.append(cov)

        if plot_env:
            pass # raw_input('CEM iter 0')

        ### remaining iterations
        for iter in xrange(1, config['iters']):
            mean, cov = self._cem_step(x0, obs, mean, cov, config['M'], config['K'], lower_bound, upper_bound,
                                       dU, T, dynamics, cost_funcs, plot_env=plot_env)
            self.means.append(mean)
            self.covs.append(cov)

            if plot_env:
                pass # raw_input('CEM iter {0}'.format(iter))

        sample = Sample(meta_data=self._meta_data, T=T)
        U = mean.reshape(T, dU)
        U[:, -1] = 0.  # TODO hack
        sample.set_U(U, t=slice(0, T))
        sample.set_O(obs, t=0)
        sample.set_X(x0, t=0)
        sample.rollout(dynamics)

        return sample

    def _cem_step(self, x0, obs, mean, cov, M, K, lower_bound, upper_bound, dU, T, dynamics, cost_funcs, plot_env=None):
        """
        :param mean: mean of trajectory controls Gaussian
        :param obs: observation
        :param cov: covariance of trajectory controls Gaussian
        :param M: sample M controls
        :param K: keep K lowest cost trajectories
        :return: mean, cov
        """
        ### sample trajectories
        M_controls = []
        while len(M_controls) < M:
            control = np.random.multivariate_normal(mean, cov)
            if np.all(control < upper_bound) and np.all(control > lower_bound):
                M_controls.append(control)
        samples = []
        for U_flat in M_controls:
            sample = Sample(meta_data=self._meta_data, T=T)
            U = U_flat.reshape(T, dU)
            U[:,-1] = 0. # TODO hack
            sample.set_U(U, t=slice(0,T))
            sample.set_O(obs, t=0)
            sample.set_X(x0, t=0)
            for sub_control, u_sub in self._config['fixed'].items():
                sample.set_U(np.array([list(u_sub) * T]).T, t=slice(0, T), sub_control=sub_control)
            sample.rollout(dynamics)
            samples.append(sample)

        ### keep lowest K cost trajectories
        costs = [0.] * len(samples)
        for cost_func in cost_funcs:
            if hasattr(cost_func, 'eval_batch'):
                costs = [c + cost_func_approx.J for c, cost_func_approx in zip(costs, cost_func.eval_batch(samples))]
            else:
                costs = [c + cost_func.eval(sample).J for c, sample in zip(costs, samples)]
        samples_costs = zip(samples, costs)
        samples_costs = sorted(samples_costs, key=lambda x: x[1])
        K_samples = [x[0] for x in samples_costs[:K]]

        ### fit Gaussian
        data = np.vstack([s.get_U().ravel() for s in K_samples])
        new_mean = np.mean(data, axis=0)
        new_cov = np.cov(data, rowvar=0)

        if plot_env:
            plot_env.rave_env.clear_plots()
            colors = [(0,1,0)]*K + [(1,0,0)]*(M-K)
            for sample, color in zip([x[0] for x in samples_costs], colors):
                for t in xrange(T-1):
                    plot_env.rave_env.plot_segment(sample.get_X(t=t, sub_state='position'),
                                                   sample.get_X(t=t+1, sub_state='position'),
                                                   color=color)

        return new_mean, new_cov





