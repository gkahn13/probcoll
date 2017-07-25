import numpy as np
import cv2
import os
from general.agent.agent import Agent
from general.state_info.sample import Sample
from robots.sim_rccar.simulation.square_env import SquareEnv
from robots.sim_rccar.simulation.square_banked_env import SquareBankedEnv
from robots.sim_rccar.simulation.cylinder_env import CylinderEnv

from config import params

class AgentSimRCcar(Agent):

    def __init__(self):
        if params['sim']['sim_env'] == 'square':
            self.env = SquareEnv(params['sim'])
        elif params['sim']['sim_env'] == 'square_banked':
            self.env = SquareBankedEnv(params['sim'])
        elif params['sim']['sim_env'] == 'cylinder':
            self.env = CylinderEnv(params['sim'])
        else:
            raise NotImplementedError(
                "Environment {0} is not valid".format(
                    params['sim']['sim_env']))

        self._curr_rollout_t = 0
        self.reset()
        self.last_n_obs = [np.zeros(params['O']['dim']) for _ in xrange(params['model']['num_O'])]  

    def sample_policy(self, policy, T=1, rollout_num=0, is_testing=False, only_noise=False):
        visualize = params['planning'].get('visualize', False)
        sample_noise = Sample(meta_data=params, T=T)
        sample_no_noise = Sample(meta_data=params, T=T)
        for t in xrange(T):
            # Get observation and act
            o_t = self.get_observation()
            self.last_n_obs.pop(0)
            self.last_n_obs.append(o_t)
            # TODO only_noise

            if is_testing:
                u_t, u_t_no_noise = policy.act(
                    self.last_n_obs,
                    t,
                    rollout_num,
                    only_noise=only_noise,
                    only_no_noise=is_testing,
                    visualize=visualize)
                self.act(u_t_no_noise)
            else:
                u_t, u_t_no_noise = policy.act(
                    self.last_n_obs,
                    self._curr_rollout_t,
                    rollout_num,
                    only_noise=only_noise,
                    visualize=visualize)
                self.act(u_t)
            # TODO possibly determine state before
            x_t = self.get_state()
            # Record
            sample_noise.set_X(x_t, t=t)
            sample_noise.set_O(o_t, t=t)
            sample_noise.set_O([int(self.coll)], t=t, sub_obs='collision')
            sample_no_noise.set_X(x_t, t=t)
            sample_no_noise.set_O(o_t, t=t)
            sample_no_noise.set_O([int(self.coll)], t=t, sub_obs='collision')

            if not is_testing:
                sample_noise.set_U(u_t, t=t)

            if not only_noise:
                sample_no_noise.set_U(u_t_no_noise, t=t)

            if self.coll:
                self._curr_rollout_t = 0
                break
            else:
                self._curr_rollout_t += 1

        return sample_noise, sample_no_noise, t 

    def reset(self, pos=None, ori=None, hard_reset=False):
        self._obs = self.env.reset(pos=pos, hpr=ori, hard_reset=hard_reset)
        self.coll = False
        if self.coll or hard_reset:
            self.last_n_obs = [np.zeros(params['O']['dim']) for _ in xrange(params['model']['num_O'])]

    def get_observation(self):
        obs_sample = Sample(meta_data=params, T=1)
        front_im, back_im = np.split(self._obs, 2, axis=2)
        if params['sim'].get('use_depth', False):
            im = AgentSimRCcar.process_depth(front_im)
            back_im = AgentSimRCcar.process_depth(back_im)
        else:
            im = AgentSimRCcar.process_image(front_im)
            back_im = AgentSimRCcar.process_image(back_im)

        obs_sample.set_O(im.ravel(), t=0, sub_obs='camera')
        obs_sample.set_O(back_im.ravel(), t=0, sub_obs='back_camera')
        return obs_sample.get_O(t=0)

    def get_state(self):
        state_sample = Sample(meta_data=params, T=1)
        pos = self._info['pos']
        ori = self._info['hpr']
        vel = [self._info['vel']]
        state_sample.set_X(pos, t=0, sub_state='position')
        state_sample.set_X(ori, t=0, sub_state='orientation')
        state_sample.set_X(vel, t=0, sub_state='velocity')
        return state_sample.get_X(t=0)

    def get_pos_ori(self):
        return self._info['pos'], self._info['hpr']

    @staticmethod
    def process_depth(image):
        im = np.reshape(image, (image.shape[0], image.shape[1]))
        model_shape = (params['O']['camera']['height'], params['O']['camera']['width'])
        if im.shape != model_shape:
            im = cv2.resize(im, model_shape, interpolation=cv2.INTER_AREA)
        return im.astype(np.uint8)

    @staticmethod
    def process_image(image):
        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

        image = rgb2gray(image)
        im = cv2.resize(
            image,
            (params['O']['camera']['height'], params['O']['camera']['width']),
            interpolation=cv2.INTER_AREA) #TODO how does this deal with aspect ratio 
        return im

    def act(self, u=[0.0, 0.0]):
        self._obs, _, self.coll, self._info = self.env.step(u)
    
    def close(self):
        pass
