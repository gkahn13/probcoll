import numpy as np

from general.algorithm.cost_probcoll import CostProbcoll

class CostProbcollPointquad(CostProbcoll):

    def plot(self, world, sample):
        rave_env = world.rave_env
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
