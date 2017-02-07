import copy
import itertools
import numpy as np

import rll_quadrotor.utility.transformations as tft
from rll_quadrotor.state_info.sample import Sample

from config import params

class Conditions(object):
    def __init__(self, cond_params=None):
        self.cond_params = cond_params if cond_params is not None else params['conditions']

        self.num_repeats = self.cond_params['repeats']
        self.randomize_conds = self.cond_params['randomize_conds']
        self.randomize_reps = self.cond_params['randomize_reps']

        self.reset()

    def reset(self):
        range_params = self.cond_params['range']
        default_params = self.cond_params['default']
        sample = Sample(T=2, meta_data=copy.copy(params))

        if self.randomize_conds:
            # fully uniform random sampling
            num_conds = np.prod([np.prod(val['num']) for val in range_params.values()])
            conds = []
            for i in xrange(num_conds):
                x = np.nan * np.zeros(params['X']['dim'])
                # set each substate as default or randomize
                for substate, val in default_params.items():
                    if substate in range_params:
                        # randomize that substate
                        val = np.random.uniform(range_params[substate]['min'],
                                                range_params[substate]['max'])
                    idxs = sample.get_X_idxs(sub_state=substate)
                    x[idxs] = val
                conds.append(x)
        else:
            # find all combinations of substates, linear spacing
            substate_combos = []
            for substate, val in range_params.items():
                lower, upper, num = val['min'], val['max'], val['num']
                ranges = [np.linspace(lower[i], upper[i], num[i]) for i in xrange(len(num))]
                combos = [(substate, r) for r in list(itertools.product(*ranges))]
                substate_combos.append(combos)
            all_combos = list(itertools.product(*substate_combos))

            # set each substate as default or the above combination
            conds = []
            for combo in all_combos:
                combo = dict(combo)
                x = np.nan * np.zeros(params['X']['dim'])
                for substate, val in default_params.items():
                    if substate in combo:
                        val = combo[substate]
                    idxs = sample.get_X_idxs(sub_state=substate)
                    x[idxs] = val

                conds.append(x)

        self.conds = conds
        self.reps = {c: dict() for c in xrange(self.length)}

    @property
    def repeats(self):
        return self.num_repeats

    @property
    def length(self):
        return len(self.conds)

    @property
    def num_test(self):
        return self.cond_params['num_test']

    def get_cond(self, cond, rep=None):
        assert(0 <= cond < self.length)

        x0 = np.copy(self.conds[cond])
        if rep is None:
            return x0

        assert(0 <= rep < self.repeats)

        if rep not in self.reps[cond]:
            if self.randomize_reps:
                sample = Sample(T=2, meta_data=copy.copy(params))

                perturb_params = self.cond_params['perturb']
                for x_substate in params['X']['order']:
                    if x_substate in perturb_params:
                        perturb = np.array(perturb_params[x_substate])
                        idxs = sample.get_X_idxs(sub_state=x_substate)

                        if x_substate == 'orientation':
                            # hardcoded conversion from rpy to quaternion
                            quat_wxyz = x0[idxs]
                            rpy_noise = tft.euler_from_quaternion(np.hstack((quat_wxyz[1:], quat_wxyz[0]))) + \
                                        np.random.uniform(-perturb, perturb)
                            quat_xyzw_noisy = tft.quaternion_from_euler(*rpy_noise)
                            x0[idxs] = np.hstack((quat_xyzw_noisy[-1], quat_xyzw_noisy[:-1]))
                        else:
                            x0[idxs] += np.random.uniform(-perturb, perturb)

            self.reps[cond][rep] = x0

        return self.reps[cond][rep]

    ##################################
    # manually add/remove conditions #
    ##################################

    def clear(self):
        self.cond_params['range'] = {}
        self.conds = []
        self.reps = {}

    def add(self, x):
        self.conds.append(x)
        self.reps[self.length-1] = dict()
