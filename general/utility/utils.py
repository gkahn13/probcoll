import time
from Queue import PriorityQueue
from collections import defaultdict

import numpy as np

def finite_differences(x, func, eps=1e-5):
    """
    :param x: input where fd evaluated at
    :type x: np.ndarray
    :param func: function func(x) outputs np.ndarray
    :return: output dim X input dim
    """
    xm, xp = np.copy(x), np.copy(x)
    J = np.zeros((len(func(x))), len(x), dtype=float)

    for i, x_i in enumerate(x):
        xp[i] = x_i + eps
        yp = func(xp)
        xp[i] = x_i

        xm[i] = x_i - eps
        ym = func(xm)
        xm[i] = x_i

        J[:,i] = (yp - ym) / (2 * eps)

    return J

def nested_max(l):
    curr_max = -np.inf
    fringe = [l]
    while len(fringe) > 0:
        popped = fringe.pop()
        if type(popped) is not list:
            curr_max = max(curr_max, popped)
        else:
            if len(popped) > 0:
                fringe.append(popped[0])
                fringe.append(popped[1:])

    return curr_max

    # if len(l) == 0:
    #     return -np.inf
    # if type(l[0]) is not list:
    #     return max(l[0], nested_max(l[1:]))
    # else:
    #     return max(nested_max(l[0]), nested_max(l[1:]))

class MyPriorityQueue(PriorityQueue):
    def __init__(self):
        PriorityQueue.__init__(self)
        self.counter = 0
        self.put_counter = 0
        self.get_counter = 0

    def put(self, item, priority):
        PriorityQueue.put(self, (priority, self.counter, item))
        self.counter += 1
        self.put_counter += 1

    def get(self, *args, **kwargs):
        priority, _, item = PriorityQueue.get(self, *args, **kwargs)
        self.get_counter += 1
        return item, priority

class TimeIt(object):
    def __init__(self, prefix=''):
        self.prefix = prefix
        self.start_times = dict()
        self.elapsed_times = defaultdict(int)

    def start(self, name):
        assert(name not in self.start_times)
        self.start_times[name] = time.time()

    def stop(self, name):
        assert(name in self.start_times)
        self.elapsed_times[name] += time.time() - self.start_times[name]
        self.start_times.pop(name)

    def elapsed(self, name):
        return self.elapsed_times[name]

    def __str__(self):
        s = ''
        names_elapsed = sorted(self.elapsed_times.items(), key=lambda x: x[1], reverse=True)
        for name, elapsed in names_elapsed:
            if 'total' not in self.elapsed_times:
                s += '{0}: {1: <10} {2:.1f}\n'.format(self.prefix, name, elapsed)
            else:
                assert(self.elapsed_times['total'] >= max(self.elapsed_times.values()))
                pct = 100. * elapsed / self.elapsed_times['total']
                s += '{0}: {1: <10} {2:.1f} ({3:.1f}%)\n'.format(self.prefix, name, elapsed, pct)
        return s
