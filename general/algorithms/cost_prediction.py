import numpy as np

from rll_quadrotor.planning.ilqr.cost.cost import Cost
from rll_quadrotor.planning.ilqr.util.approx import CostApprox

from config import params

class CostPrediction(Cost):
    def __init__(self, bootstrap, **kwargs):
        self.bootstrap = bootstrap
        self.weight = kwargs.pop('weight')
        self.eval_cost = kwargs.pop('eval_cost')
        self.pre_activation = kwargs.pop('pre_activation')

        self.probs_mean_batch = None
        self.probs_std_batch = None

    def eval(self, sample):
        raise NotImplementedError('Really old...')

        ### TODO: ignore jacobian computation b/c it is slow
        assert(min([True]+[x_meta in sample._meta_data['X']['order'] for x_meta in self.bootstrap.pred_nets[0].X_order]))
        assert(min([True]+[u_meta in sample._meta_data['U']['order'] for u_meta in self.bootstrap.pred_nets[0].U_order]))
        assert(min([True]+[o_meta in sample._meta_data['O']['order'] for o_meta in self.bootstrap.pred_nets[0].O_order]))

        T = sample._T
        xdim = sample._xdim
        udim = sample._udim

        cst_approx = CostApprox(T, xdim, udim)

        # X_idxs = self.bootstrap.pred_nets[0].X_idxs(sample._meta_data)
        # U_idxs = self.pred_net.U_idxs(sample._meta_data)
        # T_net = self.bootstrap.pred_nets[0].T

        probs_mean, probs_std = self.bootstrap.eval_sample(sample)
        # X_jac, U_jac = self.pred_net.eval_jac_sample(sample)

        # X_jac = X_jac.reshape((T_net, T_net, len(X_idxs)))
        # U_jac = U_jac.reshape((T_net, T_net-1, len(U_idxs)))

        for t in xrange(self.bootstrap.pred_nets[0].T):
            l = probs_mean[t] # + 1.0*probs_std[t] # TODO
            # lx = X_jac[t].sum(axis=0)
            # lxx = 0 # np.outer(lx, lx)
            # # if t < self.pred_net.T - 1:
            # #     lu = U_jac[t].sum(axis=0)
            # #     luu = 0 # np.outer(lu, lu)

            cst_approx.l[t] = l
            # cst_approx.lx[t, X_idxs] = lx
            # cst_approx.lxx[t][np.ix_(X_idxs, X_idxs)] = lxx
            # if t < self.pred_net.T - 1:
            #     cst_approx.lu[t, U_idxs] = lu
            #     cst_approx.luu[t][np.ix_(U_idxs, U_idxs)] = luu

        cst_approx.J = np.sum(cst_approx.l)
        cst_approx *= self.weight

        return cst_approx

    def eval_batch(self, samples,
                   speed_func=lambda s: np.linalg.norm(s.get_U(), axis=1).mean()):
        """
        If some samples are None, returns an np.inf filled CostApprox at that index
        """
        orig_samples = samples
        samples = [s for s in samples if s is not None]

        for sample in samples:
            assert(min([True]+[x_meta in sample._meta_data['X']['order'] for x_meta in self.bootstrap.X_order]))
            assert(min([True]+[u_meta in sample._meta_data['U']['order'] for u_meta in self.bootstrap.U_order]))
            assert(min([True]+[o_meta in sample._meta_data['O']['order'] for o_meta in self.bootstrap.O_order]))

        T_bootstrap = self.bootstrap.T
        T = samples[0]._T
        xdim = samples[0]._xdim
        udim = samples[0]._udim

        cst_approxs = [CostApprox(T, xdim, udim) for _ in samples]

        ### evaluate model on all time steps
        num_avg = 10 if self.bootstrap.dropout is not None else 1  # TODO
        probs_mean_batch, probs_std_batch = self.bootstrap.eval_sample_batch(samples,
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


class CostPredictionGroundTruth(Cost):
    def __init__(self, bootstrap, **kwargs):
        self.bootstrap = bootstrap
        self.world = kwargs.pop('world')
        self.weight = kwargs.pop('weight')
        self.eval_cost = kwargs.pop('eval_cost')
        self.pre_activation = kwargs.pop('pre_activation')

        self.probs_mean_batch = None
        self.probs_std_batch = None

    def eval(self, sample):
        return self.eval_batch([sample])[0]

    def eval_batch(self, samples):
        """
        If some samples are None, returns an np.inf filled CostApprox at that index
        """
        orig_samples = samples
        samples = [s for s in samples if s is not None]

        T_bootstrap = self.bootstrap.T
        T = samples[0]._T
        xdim = samples[0]._xdim
        udim = samples[0]._udim

        cst_approxs = [CostApprox(T, xdim, udim) for _ in samples]

        ### evaluate model on all time steps
        probs_mean_batch, probs_std_batch = [], []
        for sample in samples:
            is_collision = self.world.is_collision(sample, t=slice(0, T_bootstrap))
            if self.pre_activation:
                probs_mean_batch.append(np.inf if is_collision else -np.inf)
            else:
                probs_mean_batch.append(1. if is_collision else 0.)

            ### 100% confident
            probs_std_batch.append(0.)

        probs_mean_batch = np.array(probs_mean_batch).reshape((len(samples), 1))
        probs_std_batch = np.array(probs_std_batch).reshape((len(samples), 1))

        ### for recording
        self.probs_mean_batch = probs_mean_batch
        self.probs_std_batch = probs_std_batch

        for sample, cst_approx, probs_mean, probs_std in zip(samples, cst_approxs, probs_mean_batch, probs_std_batch):
            t = T_bootstrap - 1
            speed = np.linalg.norm(sample.get_U(t=t))
            sigmoid = lambda x: (1. / (1. + np.exp(-x)))

            l = eval(self.eval_cost)

            cst_approx.l[t] = l

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

