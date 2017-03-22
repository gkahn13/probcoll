import sys, termios, tty

import numpy as np
import scipy
import robots.bebop2d.ros.transformations as tft

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
    for t in xrange(sample._T):
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
    return np.array([np.sum(np.array(counts) == i) for i in xrange(num_counts)]) / float(len(counts))
