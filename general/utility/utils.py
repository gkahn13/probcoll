import sys
import time
if sys.version_info.major == 2:
    from Queue import PriorityQueue
else:
    from queue import PriorityQueue
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
        
import sys, termios, tty

import numpy as np
import scipy
import general.utility.transformations as tft

def posquat_to_pose(p, q):
    """
    :param p: position 3d array
    :param q: quaternion wxyz 4d array
    :return: 4x4 ndarray
    """
    # q_xyzw = list(q[1:]) + [q[0]]
    # pose = tft.quaternion_matrix(q_xyzw) # TODO: how does tf code fail...

    # W, X, Y, Z = q
    # xx      = X * X;
    # xy      = X * Y;
    # xz      = X * Z;
    # xw      = X * W;
    #
    # yy      = Y * Y;
    # yz      = Y * Z;
    # yw      = Y * W;
    #
    # zz      = Z * Z;
    # zw      = Z * W;
    #
    # m00  = 1 - 2 * ( yy + zz );
    # m01  =     2 * ( xy - zw );
    # m02 =     2 * ( xz + yw );
    #
    # m10  =     2 * ( xy + zw );
    # m11  = 1 - 2 * ( xx + zz );
    # m12  =     2 * ( yz - xw );
    #
    # m20  =     2 * ( xz - yw );
    # m21  =     2 * ( yz + xw );
    # m22 = 1 - 2 * ( xx + yy );
    #
    # m03  = m13 = m23 = m30 = m31 = m32 = 0;
    # m33 = 1;
    #
    # pose = np.array([[m00,m01,m02,m03],
    #                  [m10,m11,m12,m13],
    #                  [m20,m21,m22,m23],
    #                  [m30,m31,m32,m33]])

    pose = tft.quaternion_matrix(q)
    pose[:3,3] = p
    return pose



def pose_to_posquat(pose):
    """
    :param pose: 4x4 ndarray
    :return: position 3d array, quaternion wxyz 4d array
    """
    p = pose[:3,3]
    q_wxyz = tft.quaternion_from_matrix(pose)
    return p, q_wxyz

def posquats_to_poses(ps, qs):
    poses = []
    for p, q in zip(ps, qs):
        poses.append(posquat_to_pose(p, q))
    return np.array(poses)

def poses_to_posquats(poses):
    ps, qs = [], []
    for pose in poses:
        p, q = pose_to_posquat(pose)
        ps.append(p)
        qs.append(q)

    return np.array(ps), np.array(qs)

def sample_to_poses(sample):
    poses = []
    for t in range(sample._T):
        pos = sample.get_X(t=t, sub_state='position')
        quat = sample.get_X(t=t, sub_state='orientation')
        poses.append(posquat_to_pose(pos, quat))
    return poses

def init_component(params, **more_params):
    if 'args' in params:
        p = {}
        p.update(params['args'])
        p.update(more_params)
        return params['type'](**p)
    else:
        return params['type'](**more_params)

def calc_chol(X):
    try:
        return np.linalg.cholesky(X)
    except np.linalg.LinAlgError:
        raise ValueError('attempted to cholesky a non-PD matrix')


def inv_with_chol(L):
    dim = L.shape[0]
    inv = scipy.linalg.solve_triangular(
        L.T, scipy.linalg.solve_triangular(L, np.eye(dim), lower=True))
    inv_chol = np.linalg.cholesky(inv)
    return inv, inv_chol


def chol_and_inv(X):
    L = calc_chol(X)
    inv, inv_chol = inv_with_chol(L)
    return L, inv, inv_chol

class Getch:
    @staticmethod
    def getch(block=True):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

def register_callback(instance, method, callback):
    """
    Intended use:
    instance.method = register_callback(instance, method, callback)

    :param instance: class instance
    :param method: class method
    :param callback: callback, takes as argument the instance
    :return: original method that also calls the callback
    """
    orig_method = method
    def callback_method(*args):
        orig_method(*args)
        callback(instance)
    return callback_method

def to_hist(counts, num_counts):
    """
    counts contains numbers [0, num_counts)
    returns histogram
    """
    return np.array([np.sum(np.array(counts) == i) for i in range(num_counts)]) / float(len(counts))

def smooth(x, window_len, window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(x, w/w.sum(), mode='same')

    return y
