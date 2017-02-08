import numpy as np

from general.algorithm.prediction.cost_probcoll import CostProbcoll

from general.planning.cost.cost import Cost
from general.planning.cost.approx import CostApprox
from general.utility.utils import posquat_to_pose

from config import params

class CostProbcollPointquad(CostProbcoll):

    def __init__(self, bootstrap, agent, **kwargs):
        self.agent = agent # TODO
        CostProbcoll.__init__(self, bootstrap, **kwargs)

    def eval_batch(self, samples):
        """
        Hardcoded case in which the model
            (1) only depends on state X
            (2) horizon T = 1
        If above not true, calls CostProbcoll.eval_batch
        """
        return CostProbcoll.eval_batch(self, samples)

        # if self.bootstrap.T != 1:
        #     return CostProbcoll.eval_batch(self, samples)
        #
        # assert(self.bootstrap.T == 1)
        #
        # orig_samples = samples
        # samples = [s for s in samples if s is not None]
        #
        # for sample in samples:
        #     assert (min([True] + [x_meta in sample._meta_data['X']['order'] for x_meta in self.bootstrap.X_order]))
        #     assert (min([True] + [u_meta in sample._meta_data['U']['order'] for u_meta in self.bootstrap.U_order]))
        #     assert (min([True] + [o_meta in sample._meta_data['O']['order'] for o_meta in self.bootstrap.O_order]))
        #
        # T = samples[0]._T
        # xdim = samples[0]._xdim
        # udim = samples[0]._udim
        #
        # cst_approxs = [CostApprox(T, xdim, udim) for _ in samples]
        #
        # ### fill in observations for all samples at all timesteps TODO temp
        # for sample in samples:
        #     for t in xrange(T):
        #         x_t = sample.get_X(t=t)
        #         o_t = self.agent.get_observation(x_t)
        #         sample.set_O(o_t, t=t)
        #
        # ### evaluate model on all time steps
        # Xs, Us, Os = [], [], []
        # for sample in samples:
        #     for t in xrange(T):
        #         X = [[]]
        #         Xs.append(X)
        #
        #         U = sample.get_U()[t, self.bootstrap.U_idxs(params)]
        #         Us.append(U.reshape((-1, len(U))))
        #
        #         O = sample.get_O()[t, self.bootstrap.O_idxs(params)]
        #         Os.append(O.reshape((-1, len(O))))
        #
        # num_avg = 10 if self.bootstrap.dropout is not None else 1  # TODO
        # probs_mean_batch_flat, probs_std_batch_flat = self.bootstrap.eval_batch(Xs, Us, Os, num_avg=num_avg)
        # probs_mean_batch = np.array(probs_mean_batch_flat).reshape((len(samples), T))
        # probs_std_batch = np.array(probs_std_batch_flat).reshape((len(samples), T))
        #
        # ### for recording
        # self.probs_mean_batch = probs_mean_batch
        # self.probs_std_batch = probs_std_batch
        #
        # for cst_approx, probs_mean, probs_std in zip(cst_approxs, probs_mean_batch, probs_std_batch):
        #     for t in xrange(T):
        #         speed = np.linalg.norm(sample.get_U(t=t))
        #
        #         l = eval(self.eval_cost)
        #
        #         cst_approx.l[t] = l
        #
        #     cst_approx.J = np.sum(cst_approx.l)
        #     cst_approx *= self.weight
        #
        # ### push in unfilled CostApprox for None samples
        # cst_approxs_full = []
        # cst_approxs_idx = 0
        # for s_orig in orig_samples:
        #     if s_orig is not None:
        #         cst_approxs_full.append(cst_approxs[cst_approxs_idx])
        #         cst_approxs_idx += 1
        #     else:
        #         cst_inf = CostApprox(T, xdim, udim)
        #         cst_inf.J = np.inf
        #         cst_approxs_full.append(cst_inf)
        #
        # return cst_approxs_full

    def plot(self, world, sample):
        rave_env = world.env.rave_env
        cst_approx = self.eval(sample)

        pos = sample.get_X(sub_state='position')
        vel_jac = cst_approx.lx[:, sample.get_X_idxs(sub_state='linearvel')]
        probs = cst_approx.l / self.weight

        length = 0.5

        for pos, vel, prob in zip(pos, vel_jac, probs):
            if np.linalg.norm(vel) > 1e-6:
                pos_end = pos - (length * prob) * vel / np.linalg.norm(vel)
                rave_env.plot_segment(pos, pos_end, color=(1,0,1), linewidth=3.0)
                rave_env.plot_point(pos_end, color=(1,0,1), size=0.025)

                pos_end_full = pos - (length * 1.0) * vel / np.linalg.norm(vel)
                rave_env.plot_segment(pos-[0,0,0.01], pos_end_full-[0,0,0.01], color=(0,0,0), linewidth=3.0)


class CostCollisionPointquad(Cost):
    def __init__(self, env, weight, buffer, alpha):
        """
        :type env: simulation.environment.Environment
        """
        self.env = env
        self.weight = weight
        self.buffer = buffer
        self.alpha = alpha

        self.probs_mean_batch = []
        self.probs_std_batch = []

    def eval(self, sample):
        """
        Using a distance cost
        """
        T = sample._T
        xdim = sample._xdim
        udim = sample._udim

        cst_approx = CostApprox(T, xdim, udim)

        for t in xrange(T):
            pos = sample.get_X(t=t, sub_state='position')
            quat = sample.get_X(t=t, sub_state='orientation')
            pose = posquat_to_pose(pos, quat)
            speed = np.linalg.norm(sample.get_U(t=t))

            robotpos_dist_contact_pos = self.env.closest_collision(pose=pose, plot=False, contact_dist=1e4)
            if robotpos_dist_contact_pos is not None:
                _, dist, _, _ = robotpos_dist_contact_pos

                speed = 1. # TODO
                sigmoid = lambda x: (1. / (1. + np.exp(-x)))
                cst_approx.l[t] = (1. / float(T)) * speed * sigmoid(self.alpha * (self.buffer - dist))

        cst_approx.J = np.sum(cst_approx.l)
        cst_approx *= self.weight

        return cst_approx
