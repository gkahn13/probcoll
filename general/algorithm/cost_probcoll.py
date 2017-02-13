import numpy as np

from general.planning.cost.cost import Cost
from general.planning.cost.approx import CostApprox

from config import params

class CostProbcoll(Cost):
    def __init__(self, probcoll_model):
        self._probcoll_model = probcoll_model

        cost_probcoll_params = params['probcoll']['cost']
        self.weight = cost_probcoll_params['weight']
        self.eval_cost = cost_probcoll_params['eval_cost']
        self.pre_activation = cost_probcoll_params['pre_activation']

        self.probs_mean_batch = None
        self.probs_std_batch = None

    def eval(self, sample, speed_func=lambda s: np.linalg.norm(s.get_U(), axis=1).mean()):
        return self.eval_batch([sample], speed_func=speed_func)[0]

    def eval_batch(self, samples, speed_func=lambda s: np.linalg.norm(s.get_U(), axis=1).mean()):
        """
        If some samples are None, returns an np.inf filled CostApprox at that index
        """
        orig_samples = samples
        samples = [s for s in samples if s is not None]

        for sample in samples:
            assert(min([True]+[x_meta in sample._meta_data['X']['order'] for x_meta in self._probcoll_model.X_order]))
            assert(min([True]+[u_meta in sample._meta_data['U']['order'] for u_meta in self._probcoll_model.U_order]))
            assert(min([True]+[o_meta in sample._meta_data['O']['order'] for o_meta in self._probcoll_model.O_order]))

        T_bootstrap = self._probcoll_model.T
        T = samples[0]._T
        xdim = samples[0]._xdim
        udim = samples[0]._udim

        cst_approxs = [CostApprox(T, xdim, udim) for _ in samples]

        ### evaluate model on all time steps
        num_avg = 10 if self._probcoll_model.dropout is not None else 1  # TODO
        probs_mean_batch, probs_std_batch = self._probcoll_model.eval_sample_batch(samples,
                                                                             num_avg=num_avg,
                                                                             pre_activation=self.pre_activation)
        probs_mean_batch = np.array(probs_mean_batch).ravel()
        probs_std_batch = np.array(probs_std_batch).ravel()

        ### for recording
        self.probs_mean_batch = probs_mean_batch
        self.probs_std_batch = probs_std_batch

        for sample, cst_approx, probs_mean, probs_std in zip(samples, cst_approxs, probs_mean_batch, probs_std_batch):
            t = T_bootstrap - 1
            speed = speed_func(sample)
            sigmoid = lambda x: (1. / (1. + np.exp(-x)))

            l = eval(self.eval_cost)
            cst_approx.l[0] = l

            cst_approx.J = np.sum(cst_approx.l)
            cst_approx *= self.weight

        ### push in unfilled CostApprox for None samples
        cst_approxs_full = []
        cst_approxs_idx = 0
        for s_orig in orig_samples:
            if s_orig is not None:
                cst_approxs_full.append(cst_approxs[cst_approxs_idx])
                cst_approxs_idx += 1
            else:
                cst_inf = CostApprox(T, xdim, udim)
                cst_inf.J = np.inf
                cst_approxs_full.append(cst_inf)

        return cst_approxs_full

    def plot(self, world, sample):
        pass
