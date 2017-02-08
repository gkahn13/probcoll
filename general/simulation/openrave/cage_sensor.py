import numpy as np

import openravepy as rave

class CageSensor:
    """
    Short-range cage of depth sensors
    """
    def __init__(self, rave_env, num_rays, max_range):
        """
        :param rave_env: RaveEnv
        :param num_rays: number of rays
        :param max_range: meters
        """
        self.rave_env = rave_env
        self.num_rays = num_rays
        self.max_range = max_range

        self.collision_checker = rave.RaveCreateCollisionChecker(self.rave_env.env, 'ode')
        self.collision_checker.SetCollisionOptions(0)

    def get_directions(self, origin):
        """
        :return: N x 3 array where each row is the unit vector direction of the sensor
        """
        origin_rot = origin[:3,:3]

        theta = np.linspace(-np.pi, np.pi, self.num_rays//2) # xy plane
        psi = np.linspace(-np.pi, np.pi, self.num_rays - self.num_rays//2) # yz plane

        x_theta = self.max_range * np.cos(theta)
        y_theta = self.max_range * np.sin(theta)
        z_theta = self.max_range * np.zeros(len(theta))

        x_psi = self.max_range * np.zeros(len(psi))
        y_psi = self.max_range * np.cos(psi)
        z_psi = self.max_range * np.sin(psi)

        x = np.hstack((x_theta, x_psi))
        y = np.hstack((y_theta, y_psi))
        z = np.hstack((z_theta, z_psi))

        xyz = np.vstack((x.ravel(),y.ravel(),z.ravel())).T
        xyz_origin = origin_rot.dot(xyz.T).T

        # xyz_origin = xyz_origin.reshape((dim, 3))
        dir = (xyz_origin.T / np.linalg.norm(xyz_origin, axis=1)).T

        return dir

    def read(self, origin, plot=False):
        orig_pose = self.rave_env.robot.GetTransform()
        offset = np.eye(4); offset[:3, 3] = [0, 0, -1e3]
        self.rave_env.robot.SetTransform(origin.dot(offset))

        if plot:
            self.rave_env.clear_plots()

        self.collision_checker = rave.RaveCreateCollisionChecker(self.rave_env.env, 'ode')
        self.collision_checker.SetCollisionOptions(0)

        origin_pos = origin[:3,3]
        origin_rot = origin[:3,:3]

        theta = np.linspace(-np.pi, np.pi, self.num_rays//2) # xy plane
        psi = np.linspace(-np.pi, np.pi, self.num_rays - self.num_rays//2) # yz plane

        x_theta = self.max_range * np.cos(theta)
        y_theta = self.max_range * np.sin(theta)
        z_theta = self.max_range * np.zeros(len(theta))

        x_psi = self.max_range * np.zeros(len(psi))
        y_psi = self.max_range * np.cos(psi)
        z_psi = self.max_range * np.sin(psi)

        x = np.hstack((x_theta, x_psi))
        y = np.hstack((y_theta, y_psi))
        z = np.hstack((z_theta, z_psi))

        xyz = np.vstack((x.ravel(),y.ravel(),z.ravel())).T
        xyz_origin = origin_rot.dot(xyz.T).T
        rays = np.hstack((np.tile(origin_pos, (xyz.shape[0],1)), xyz_origin))

        with self.rave_env.env:
            is_hits, hits = self.collision_checker.CheckCollisionRays(rays, None) # None == check all kinbodies in env

        dim = len(is_hits)
        assert(dim == self.num_rays)
        hits = hits.reshape((dim, 6))
        xyz_origin = xyz_origin.reshape((dim, 3))
        zpoints = np.zeros((dim, 3))
        zbuffer = np.zeros(dim)
        
        for i in xrange(dim):
                if is_hits[i]:
                    zpoints[i,:] = hits[i,:3]
                else:
                    zpoints[i,:] = origin_pos + xyz_origin[i,:]

                dist = np.linalg.norm(zpoints[i,:] - origin_pos)
                if dist > self.max_range:
                    zpoints[i,:] = (self.max_range/dist)*(zpoints[i,:] - origin_pos) + origin_pos

                zbuffer[i] = np.linalg.norm(zpoints[i,:] - origin_pos)

                if plot:
                    self.rave_env.plot_segment(origin_pos, zpoints[i,:], color=(1,0,0))
                    self.rave_env.plot_point(zpoints[i,:], color=(0,1,0), size=0.05)
                    # plot normal
                    if is_hits[i]:
                        p0 = hits[i,:3]
                        p1 = p0 + 0.1 * hits[i,3:]
                        self.rave_env.plot_segment(p0, p1, color=(1,1,0))

        self.rave_env.robot.SetTransform(orig_pose)

        return zbuffer


