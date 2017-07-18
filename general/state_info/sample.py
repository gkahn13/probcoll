import os.path
import numpy as np
from general.utility.utils import posquats_to_poses

from config import params as meta_data

class Sample(object):
    """ Handles representation of a trajectory (states, controls, observations) """

    def __init__(self, **kwargs):
        self._meta_data = kwargs.get('meta_data', meta_data)
        self._xdim = self._meta_data['X']['dim']
        self._udim = self._meta_data['U']['dim']
        self._odim = self._meta_data['O']['dim']
        self._T = kwargs['T'] if 'T' in kwargs else self._meta_data['T']

        self._X = kwargs.get('X', np.full((self._T, self._xdim), np.nan, dtype=np.float32))
        self._U = kwargs.get('U', np.full((self._T, self._udim), np.nan, dtype=np.float32))
        self._O = kwargs.get('O', np.full((self._T, self._odim), np.nan, dtype=np.float32))

        assert(self._X.shape == (self._T, self._xdim))
        assert(self._U.shape == (self._T, self._udim))
        assert(self._O.shape == (self._T, self._odim))

    def set(self, xuo, val, t, sub_state=None):
        if xuo == 'X':
            return self.set_X(val, t, sub_state=sub_state)
        elif xuo == 'U':
            return self.set_U(val, t)
        elif xuo == 'O':
            return self.set_O(val, t, sub_obs=sub_state)

    def set_X(self, x, t, sub_state=None):
        idxs = self.get_X_idxs(sub_state)
        assert(np.array(x).shape[-1] == idxs.stop - idxs.start)
        self._X[t,idxs] = np.copy(x)

    def set_U(self, u, t, sub_control=None):
        idxs = self.get_U_idxs(sub_control)
        assert(np.array(u).shape[-1] == idxs.stop - idxs.start)
        self._U[t,idxs] = np.copy(u)

    def set_O(self, o, t, sub_obs=None):
        idxs = self.get_O_idxs(sub_obs)
        assert(np.array(o).shape[-1] == idxs.stop - idxs.start)
        self._O[t,idxs] = np.copy(o)

    def get_X(self, t=None, sub_state=None):
        idxs = self.get_X_idxs(sub_state)
        if t is None:
            return np.copy(self._X[:,idxs])
        else:
            return self._X[t,idxs]

    def get(self, xuo, t=None):
        if xuo == 'X':
            return self.get_X(t=t)
        elif xuo == 'U':
            return self.get_U(t=t)
        elif xuo == 'O':
            return self.get_O(t=t)

    def get_U(self, t=None, sub_control=None):
        idxs = self.get_U_idxs(sub_control)
        if t is None:
            return np.copy(self._U[:,idxs])
        else:
            return self._U[t,idxs]

    def get_O(self, t=None, sub_obs=None):
        idxs = self.get_O_idxs(sub_obs)
        if t is None:
            return np.copy(self._O[:,idxs])
        else:
            return self._O[t,idxs]

    def empty_like(self):
        return Sample(meta_data=self._meta_data, T=self._T)

    def check_constraints(self, with_buffer=False):
        return True # TODO:

    def clip_U(self):
        pass

    def ros_publish(self, pub):
        """
        :param pub: rospy.Publisher PoseArray
        :return:
        """
        import rospy
        import geometry_msgs.msg as gm
        pa = gm.PoseArray()
        pa.header.frame_id = 'world'
        pa.header.stamp = rospy.Time.now()

        for t in xrange(self._T):
            pos = self.get_X(t=t, sub_state='position')
            quat_wxyz = self.get_X(t=t, sub_state='orientation')

            pose = gm.Pose()

            pose.position.x = pos[0]
            pose.position.y = pos[1]
            pose.position.z = pos[2]
            pose.orientation.w = quat_wxyz[0]
            pose.orientation.x = quat_wxyz[1]
            pose.orientation.y = quat_wxyz[2]
            pose.orientation.z = quat_wxyz[3]

            pa.poses.append(pose)

        pub.publish(pa)

    def plot_rave(self, rave_env, t=None, color=(1,0,0), scale=0.4):
        if t is None: t = slice(0, self._T)
        positions = self.get_X(t=t, sub_state='position')
        orientations = self.get_X(t=t, sub_state='orientation')

        if len(positions.shape) == 1: positions = np.array([positions])
        if len(orientations.shape) == 1: orientations = np.array([orientations])

        poses = posquats_to_poses(positions, orientations)
        rave_env.robot.SetTransform(poses[0])
        for pose in poses:
            rave_env.plot_arrow(pose, color=color, s=scale)

    def get_X_dim(self, sub_state):
        x_idxs = self.get_X_idxs(sub_state=sub_state)
        return x_idxs.stop - x_idxs.start


    def get_X_idxs(self, sub_state=None):
        if sub_state is None: return slice(0, self._xdim)
        assert(sub_state in self._meta_data['X'])
        start = self._meta_data['X'][sub_state]['idx']
        dim = self._meta_data['X'][sub_state]['dim']
        return slice(start, start+dim)

    def get_U_dim(self, sub_control):
        u_idxs = self.get_U_idxs(sub_control=sub_control)
        return u_idxs.stop - u_idxs.start

    def get_U_idxs(self, sub_control=None):
        if sub_control is None: return slice(0, self._udim)

        assert(sub_control in self._meta_data['U'])
        start = self._meta_data['U'][sub_control]['idx']
        dim = self._meta_data['U'][sub_control]['dim']
        return slice(start, start+dim)

    def get_O_dim(self, sub_obs):
        o_idxs = self.get_O_idxs(sub_obs=sub_obs)
        return o_idxs.stop - o_idxs.start

    def get_O_idxs(self, sub_obs=None):
        if sub_obs is None: return slice(0, self._odim)

        assert(sub_obs in self._meta_data['O'])
        start = self._meta_data['O'][sub_obs]['idx']
        dim = self._meta_data['O'][sub_obs]['dim']
        return slice(start, start+dim)

    def create_x(self, x_dict):
        x = np.nan * np.ones(self._xdim, dtype=np.float32)

        sub_states = self._meta_data['X']['order']
        for sub_state, val in x_dict.items():
            if sub_state in sub_states:
                x[self.get_X_idxs(sub_state)] = val

        return x

    def match(self, time_slice):
        """
        :type time_slice: slice
        :return: Sample
        """
        T = time_slice.stop - time_slice.start
        sample = Sample(meta_data=self._meta_data, T=T)
        sample.set_X(self.get_X(t=time_slice), slice(0,T))
        sample.set_U(self.get_U(t=time_slice), slice(0,T))
        sample.set_O(self.get_O(t=time_slice), slice(0,T))

        return sample

    def copy(self):
        return self.match(slice(0, self._T))

    def isfinite(self):
        return np.isfinite(self.get_X()).all() and \
               np.isfinite(self.get_U()).all() and \
               np.isfinite(self.get_O()).all()

    def rollout(self, dynamics):
        """
        Assuming x0 and all U are set, rollout sample according to dynamics
        """
        assert(np.isfinite(self.get_X(t=0).all()))
        assert(np.isfinite(self.get_U().all()))
        for t in xrange(self._T - 1):
            x_t = self.get_X(t=t)
            u_t = self.get_U(t=t)
            x_tp1 = dynamics.evolve(x_t, u_t)
            self.set_X(x_tp1, t=t+1)

    @staticmethod
    def save(fname, samples):
        assert(os.path.splitext(fname)[-1] == '.npz')
        meta_datas = [s._meta_data for s in samples]
        Xs = [s.get_X() for s in samples]
        Us = [s.get_U() for s in samples]
        Os = [s.get_O() for s in samples]
        np.savez(fname, meta_datas=meta_datas, Xs=Xs, Us=Us, Os=Os)

    @staticmethod
    def load(fname):
        assert(os.path.exists(fname))
        samples = []
        tot_T = 0
        assert (os.path.splitext(fname)[-1] == '.npz')
        d = np.load(fname)
        for meta_data, X, U, O in zip(d['meta_datas'], d['Xs'], d['Us'], d['Os']):
            T = len(X)
            tot_T += T
            s = Sample(meta_data=meta_data, T=T, X=X, U=U, O=O)
            samples.append(s)

        return samples, tot_T
