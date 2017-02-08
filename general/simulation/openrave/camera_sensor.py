from collections import defaultdict

import numpy as np

import openravepy as rave

import rll_quadrotor.utility.transformations as tft
import rll_quadrotor.utility.utils as utils

class CameraSensor:
    def __init__(self, rave_env, height, width, range, KK=None, F=0.01):
        """
        :param rave_env  : RaveEnv
        :param height    : camera pixel height
        :param width     : camera pixel width
        :param range     : max range of sensor
        :param KK        : camera intrinsics matrix
        :param F         : focal distance (meters)
        """
        self.rave_env = rave_env
        if KK is None:
             KK = np.array([[max(height,width)/2., 0, height/2.],
                            [0., max(height,width)/2., width/2.],
                            [0., 0., 1.0]])
        self.KK = KK # camera intrinsic matrix
        self.height = height # in pixels
        self.width = width # in pixels
        self.F = F # focal length (distance from origin to image plane) in meters
        self.range = range

        f = KK[0,0] # pixel focal length
        self.H = F*(height/f) # actual height of image plane in meters
        self.W = F*(width/f) # actual width of image plane in meters

        self.collision_checker = rave.RaveCreateCollisionChecker(self.rave_env.env, 'ode')
        self.collision_checker.SetCollisionOptions(0)

    def read(self, origin, light_dir, plot=False):
        """
        Returns depth reading (zbuffer)

        :param origin: 4x4 np.ndarray pose
        :param light_dir : direction of light (3d vector)
        :return np.ndarray intensity
        """
        if plot:
            self.rave_env.clear_plots()

        dirs = self._directions(origin)

        origin_world_pos = origin[:3,3]
        rays = np.hstack((np.tile(origin_world_pos, (dirs.shape[0],1)), dirs))

        with self.rave_env.env:
            is_hits, hits = self.collision_checker.CheckCollisionRays(rays, None) # None == check all kinbodies in env

        is_hits = is_hits.reshape((self.width,self.height))
        hits = hits.reshape((self.width,self.height,6))
        intensity = np.zeros((self.width,self.height))

        for i in xrange(self.height):
            for j in xrange(self.width):
                if is_hits[j,i]:
                    normal = hits[j,i,3:]
                    norm = np.linalg.norm(normal) * np.linalg.norm(light_dir)
                    # theta = np.abs(np.arccos(normal.dot(-light_dir) / norm))
                    # intensity[j,i] = theta if theta < np.pi/2. else 0
                    intensity[j,i] = max(0, normal.dot(-light_dir) / norm)
                else:
                    intensity[j,i] = 0


        dirs = dirs.reshape((self.width,self.height,3))
        zpoints = np.zeros((self.width,self.height,3))

        for i in xrange(self.height):
            for j in xrange(self.width):
                if is_hits[j,i]:
                    zpoints[j,i,:] = hits[j,i,:3]
                else:
                    zpoints[j,i,:] = origin_world_pos + dirs[j,i,:]

                dist = np.linalg.norm(zpoints[j,i,:] - origin_world_pos)
                if dist > self.range:
                    zpoints[j,i,:] = (self.range/dist)*(zpoints[j,i,:] - origin_world_pos) + origin_world_pos

                if plot:
                    self.rave_env.plot_segment(origin_world_pos, zpoints[j,i,:], color=(1,0,0))
                    self.rave_env.plot_point(zpoints[j,i,:], color=(0,1,0))

        return intensity.T

    def _directions(self, origin):
        """
        Returns rays that emanate from the origin through the image plane
        (in 'world' frame)

        :param origin: 4x4 np.ndarray pose
        """
        N = self.width*self.height

        height_offsets = np.linspace(-self.H/2.0, self.H/2.0, self.height)
        width_offsets = np.linspace(-self.W/2.0, self.W/2.0, self.width)

        height_grid, width_grid = np.meshgrid(height_offsets, width_offsets)

        height_grid_vec = height_grid.reshape((N,1))
        width_grid_vec = width_grid.reshape((N,1))
        z_vec = np.zeros((N,1))

        offsets = np.hstack((z_vec, width_grid_vec, height_grid_vec))
        points_cam = (self.range/self.F)*(np.tile(np.array([self.F,0,0]), (N,1)) + offsets)

        ref_from_world = origin
        origin_world_pos = ref_from_world[:3,3]

        directions = np.zeros((N,3))

        p_cam = np.eye(4)
        for i in xrange(N):
            p_cam[:3,3] = points_cam[i,:]
            p_world_pos = np.dot(ref_from_world, p_cam)[0:3,3]

            direction = np.array(p_world_pos) - np.array(origin_world_pos)
            directions[i,:] = direction

        return directions

    def interact(self, light_dir, init_pose=np.eye(4), step=0.1, radstep=0.1):
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

        import matplotlib.pyplot as plt
        plt.figure()

        while ch != 'q':
            self.rave_env.clear_plots()
            intensity = self.read(np.copy(pose), light_dir, plot=True)

            plt.imshow(intensity, 'gray')
            plt.show(block=False)
            plt.pause(0.01)

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
