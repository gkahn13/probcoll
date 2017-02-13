import numpy as np

from general.agent.agent import Agent
from general.state_info.sample import Sample

from general.simulation.openrave.depth_sensor import DepthSensor
from general.simulation.openrave.cage_sensor import CageSensor
from general.simulation.openrave.signed_distance_sensor import SignedDistanceSensor
from general.utility.utils import posquat_to_pose

from config import params

class AgentPointquad(Agent):
    def __init__(self, world, dynamics, obs_noise=False, dyn_noise=False):
        self._world = world
        self._dynamics = dynamics
        self.meta_data = params
        self.obs_noise = obs_noise
        self.dyn_noise = dyn_noise

        self.depth_sensor = None
        if 'laserscan' in self.meta_data['O']:
            self.depth_sensor = DepthSensor(self._world.rave_env,
                                            self.meta_data['O']['laserscan']['fov'],
                                            1.,  # arbitrary
                                            self.meta_data['O']['laserscan']['dim'],
                                            1,
                                            self.meta_data['O']['laserscan']['range'])

        self.camera = None
        if 'camera' in self.meta_data['O']:
            self.camera = self._world.panda_env

        self.cage_sensor = None
        if 'cage' in self.meta_data['O']:
            self.cage_sensor = CageSensor(self._world.rave_env,
                                          self.meta_data['O']['cage']['dim'],
                                          self.meta_data['O']['cage']['range'])

        self.sd_sensor = None
        if 'signed_distance' in self.meta_data['O']:
            self.sd_sensor = SignedDistanceSensor(self._world.rave_env,
                                                  self.meta_data['O']['signed_distance']['extents'],
                                                  self.meta_data['O']['signed_distance']['sizes'],
                                                  self.meta_data['O']['signed_distance']['max_dist'])

    def sample_policy(self, x0, policy, T=None, **policy_args):
        """
        Run the policy and collect the trajectory data

        :param x0: initial state
        :param policy: to execute
        :param policy_args: e.g. ref_traj, noise, etc
        :rtype Sample
        """
        if T is None:
            T = policy._T
        policy_sample = Sample(meta_data=self.meta_data, T=T)

        policy_sample.set_X(x0, t=0)
        for t in xrange(T):
            # get observation and act
            x_t = policy_sample.get_X(t=t)
            o_t = self.get_observation(x_t)
            u_t = policy.act(x_t, o_t, t, **policy_args)
            if self.dyn_noise:
                noise = []
                for dyn_noise_i in self.dyn_noise:
                    if dyn_noise_i > 0:
                        noise.append(np.random.normal(0., dyn_noise_i))
                    else:
                        noise.append(0)
                u_t += noise

            # record
            policy_sample.set_X(x_t, t=t)
            policy_sample.set_O(o_t, t=t)
            policy_sample.set_U(u_t, t=t)

            # propagate dynamics
            if t < T-1:
                x_tp1 = self._dynamics.evolve(x_t, u_t)
                policy_sample.set_X(x_tp1, t=t+1)

        return policy_sample

    def reset(self, x):
        """
        Reset the simulated environment as the specified state.
        Return the actual model state vector.

        :param x: state vector to reset to
        :rtype: np.ndarray
        """
        return np.copy(x)

    def get_observation(self, x, noise=True):
        """ Observation model """
        obs_sample = Sample(meta_data=self.meta_data, T=2)
        for sub_state in self.meta_data['X']['order']:
            idxs = obs_sample.get_X_idxs(sub_state=sub_state)
            obs_sample.set_X(x[idxs], t=0, sub_state=sub_state)
        for sub_state in [s for s in self.meta_data['X']['order'] if s in self.meta_data['O']['order']]:
            idxs = obs_sample.get_X_idxs(sub_state=sub_state)
            obs_sample.set_O(x[idxs], t=0, sub_obs=sub_state)

        x_pos = x[obs_sample.get_X_idxs(sub_state='position')]
        x_ori = x[obs_sample.get_X_idxs(sub_state='orientation')]
        origin = posquat_to_pose(x_pos, x_ori)

        # hacky way to make sure sensors don't see robot

        if 'collision' in self.meta_data['O']:
            is_collision = self._world.is_collision(obs_sample, t=0)
            obs_sample.set_O([float(is_collision)], t=0, sub_obs='collision')
        if self.depth_sensor:
            zbuffer = self.depth_sensor.read(origin, plot=False).ravel()
            if noise:
                zbuffer += np.random.normal(0., 0.05, len(zbuffer))
            obs_sample.set_O(zbuffer, t=0, sub_obs='laserscan')

        if self.camera:
            self.camera.set_camera_pose(origin)
            im = self.camera.get_camera_image(grayscale=True)
            if self.obs_noise:
                std = self.meta_data['O']['camera'].get('noise', None)
                if std is not None:
                    im += np.random.normal(0., std, im.shape)
                    im = np.clip(im, 0., 1.)

            obs_sample.set_O(im.ravel(), t=0, sub_obs='camera')

        if self.cage_sensor:
            cage_depths = self.cage_sensor.read(origin, plot=False).ravel()
            if noise:
                cage_depths += np.random.normal(0, 0.01, len(cage_depths))
            obs_sample.set_O(cage_depths, t=0, sub_obs='cage')

        if self.sd_sensor:
            sd = self.sd_sensor.read(origin, noise=noise).ravel()
            obs_sample.set_O(sd, t=0, sub_obs='signed_distance')

        obs = obs_sample.get_O(t=0)
        assert (np.isfinite(obs).all())

        return obs

