from collections import defaultdict

import numpy as np

import openravepy as rave

import rll_quadrotor.utility.transformations as tft
import rll_quadrotor.utility.utils as utils

class DepthSensor:
    def __init__(self, rave_env, fov_theta, fov_phi, rays_theta, rays_phi, max_range):
        """
        :param rave_env: RaveEnv
        :param fov_theta: radians horizontal
        :param fov_phi: radians vertical
        :param rays_theta: rays horizontal
        :param rays_phi: rays vertical
        :param max_range: meters
        :return:
        """
        self.rave_env = rave_env
        self.fov_theta = fov_theta
        self.fov_phi = fov_phi
        self.rays_theta = rays_theta
        self.rays_phi = rays_phi
        self.max_range = max_range

        self.collision_checker = rave.RaveCreateCollisionChecker(self.rave_env.env, 'ode')
        self.collision_checker.SetCollisionOptions(0)

    def read(self, origin, plot=False):
        if plot:
            self.rave_env.clear_plots()

        self.collision_checker = rave.RaveCreateCollisionChecker(self.rave_env.env, 'ode')
        self.collision_checker.SetCollisionOptions(0)

        origin_pos = origin[:3,3]
        origin_rot = origin[:3,:3]

        theta = 0 if self.rays_theta == 1 else np.linspace(-self.fov_theta/2., self.fov_theta/2., self.rays_theta)
        phi = np.pi/2. if self.rays_phi == 1 else np.linspace(np.pi/2.-self.fov_phi/2., np.pi/2.+self.fov_phi/2., self.rays_phi)

        theta_grid, phi_grid = np.meshgrid(theta, phi)

        x = self.max_range * np.sin(phi_grid) * np.cos(theta_grid)
        y = self.max_range * np.sin(phi_grid) * np.sin(theta_grid)
        z = self.max_range * np.cos(phi_grid)


        xyz = np.vstack((x.ravel(),y.ravel(),z.ravel())).T
        xyz_origin = origin_rot.dot(xyz.T).T
        rays = np.hstack((np.tile(origin_pos, (xyz.shape[0],1)), xyz_origin))

        with self.rave_env.env:
            is_hits, hits = self.collision_checker.CheckCollisionRays(rays, None) # None == check all kinbodies in env

        is_hits = is_hits.reshape((self.rays_phi, self.rays_theta))
        hits = hits.reshape((self.rays_phi, self.rays_theta,6))
        xyz_origin = xyz_origin.reshape((self.rays_phi, self.rays_theta,3))
        zpoints = np.zeros((self.rays_phi, self.rays_theta,3))
        zbuffer = np.zeros((self.rays_phi, self.rays_theta))

        for j in xrange(self.rays_phi):
            for i in xrange(self.rays_theta):
                if is_hits[j,i]:
                    zpoints[j,i,:] = hits[j,i,:3]
                else:
                    zpoints[j,i,:] = origin_pos + xyz_origin[j,i,:]

                dist = np.linalg.norm(zpoints[j,i,:] - origin_pos)
                if dist > self.max_range:
                    zpoints[j,i,:] = (self.max_range/dist)*(zpoints[j,i,:] - origin_pos) + origin_pos

                zbuffer[j,i] = np.linalg.norm(zpoints[j,i,:] - origin_pos)

                if plot:
                    self.rave_env.plot_segment(origin_pos, zpoints[j,i,:], color=(1,0,0))
                    self.rave_env.plot_point(zpoints[j,i,:], color=(0,1,0), size=0.05)
                    # plot normal
                    if is_hits[j,i]:
                        p0 = hits[j,i,:3]
                        p1 = p0 + 0.1 * hits[j,i,3:]
                        self.rave_env.plot_segment(p0, p1, color=(1,1,0))

        return zbuffer

    def interact(self, init_pose=np.eye(4), step=0.1, radstep=0.1):
        pose = init_pose
        ch = None

        mapping = {
            'w' : np.array([step, 0, 0, 0, 0, 0]),
            'x' : np.array([-step, 0, 0, 0, 0, 0]),
            'd' : np.array([0, step, 0, 0, 0, 0]),
            'a' : np.array([0, -step, 0, 0, 0, 0]),
            '+' : np.array([0, 0, step, 0, 0, 0]),
            '-' : np.array([0, 0, -step, 0, 0, 0]),
            'p' : np.array([0, 0, 0, radstep, 0, 0]),
            'o' : np.array([0, 0, 0, -radstep, 0, 0]),
            'l' : np.array([0, 0, 0, 0, radstep, 0]),
            'k' : np.array([0, 0, 0, 0, -radstep, 0]),
            'm' : np.array([0, 0, 0, 0, 0, radstep]),
            'n' : np.array([0, 0, 0, 0, 0, -radstep]),
        }
        mapping = defaultdict(lambda: np.zeros(6), mapping)

        while ch != 'q':
            self.rave_env.clear_plots()
            self.read(np.copy(pose), plot=True)

            ch = utils.Getch.getch()
            pos = pose[:3, 3]
            rpy = tft.euler_from_matrix(pose)

            pos += mapping[ch][:3]
            rpy += mapping[ch][3:]

            pose = tft.euler_matrix(*rpy)
            pose[:3, 3] = pos

            print('pos: {0:.2f}, {1:.2f}, {2:.2f}'.format(*list(pos)))
            print('rpy: {0:.2f}, {1:.2f}, {2:.2f}'.format(*list(rpy)))
            print('quaternion: {0}\n'.format(tft.quaternion_from_matrix(pose)))

