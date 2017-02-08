import numpy as np

class SignedDistanceSensor:
    """ Signed distance sensor """
    def __init__(self, rave_env, extents, sizes, max_dist):
        """
        :param rave_env:
        :param extents: 3d list
        :param resolutions: 3d list
        :param max_dist: max value of signed distance
        """
        self.rave_env = rave_env
        self.extents = extents
        self.sizes = sizes
        self.max_dist = max_dist

        self.xs = np.linspace(-self.extents[0]/2., self.extents[0]/2., self.sizes[0])
        self.ys = np.linspace(-self.extents[1]/2., self.extents[1]/2., self.sizes[1])
        self.zs = np.linspace(-self.extents[2]/2., self.extents[2]/2., self.sizes[2])

        self.resolutions = np.array([e / s for e, s in zip(self.extents, self.sizes)])

    def read(self, origin, noise=False):
        orig_pose = self.rave_env.robot.GetTransform()
        offset = np.eye(4); offset[:3, 3] = [0, 0, -1e3]
        self.rave_env.robot.SetTransform(origin.dot(offset))

        grid = np.zeros(list(self.sizes), dtype=float)
        for i, x in enumerate(self.xs):
            for j, y in enumerate(self.ys):
                for k, z in enumerate(self.zs):
                    pose = np.copy(origin)
                    pose[:3,3] += [x, y, z]
                    cc = self.rave_env.closest_collision(pose=pose, plot=False, contact_dist=self.max_dist)
                    if cc is None:
                        dist = self.max_dist
                    else:
                        dist = cc[1]

                    if noise:
                        dist += np.random.normal(0, 0.1*np.linalg.norm(self.resolutions))

                    grid[i, j, k] = dist

        self.rave_env.robot.SetTransform(orig_pose)

        return grid
