import numpy as np
import cv2
import os
from general.agent.agent import Agent
from general.state_info.sample import Sample
from robots.rccar.env.rccar_env import RCcarEnv

from config import params

class AgentRCcar(Agent):

    def __init__(self):
        self._env = RCcarEnv(params['env'])
        self._curr_rollout_t = 0
        self._done = False
        self.last_n_obs = [np.zeros(params['O']['dim']) for _ in range(params['model']['num_O'])]  
        self.reset()

    def sample_policy(self, policy, T=1, time_step=0, is_testing=False, only_noise=False):
        visualize = params['planning'].get('visualize', False)
        need_to_collect_sample=True
        while need_to_collect_sample:
            sample_noise = Sample(meta_data=params, T=T)
            sample_no_noise = Sample(meta_data=params, T=T)
            for t in range(T):
                # Get observation and act
                o_t = self.get_observation()
                self.last_n_obs.pop(0)
                self.last_n_obs.append(o_t)
                if is_testing:
                    u_t, u_t_no_noise = policy.act(
                        self.last_n_obs,
                        t,
                        time_step=time_step + t,
                        only_noise=only_noise,
                        only_no_noise=is_testing,
                        visualize=visualize)
                    self.act(u_t_no_noise)
                else:
                    u_t, u_t_no_noise = policy.act(
                        self.last_n_obs,
                        self._curr_rollout_t,
                        time_step=time_step + t,
                        only_noise=only_noise,
                        visualize=visualize)
                    self.act(u_t)
                # TODO possibly determine state before
                x_t = self.get_state()
                coll = self.get_coll()
                # Record
                sample_noise.set_X(x_t, t=t)
                sample_noise.set_O(o_t, t=t)
                sample_noise.set_O([coll], t=t, sub_obs='collision')
                sample_no_noise.set_X(x_t, t=t)
                sample_no_noise.set_O(o_t, t=t)
                sample_no_noise.set_O([coll], t=t, sub_obs='collision')

                if not is_testing:
                    sample_noise.set_U(u_t, t=t)

                if not only_noise:
                    sample_no_noise.set_U(u_t_no_noise, t=t)

                if self._done:
                    self._curr_rollout_t = 0
                    self.reset()
                    break
                else:
                    self._curr_rollout_t += 1
            if params['env']['check_rollouts']:
                if self._done:
                    self._logger.info('Trajectory finished')
                self.reset()
                text = input('Enter n if error during rollout, and anything else to save rollout')
                if 'n' in text:
                    need_to_collect_sample = True
                else:
                    need_to_collect_sample = False
                self.reset()
            else:
                need_to_collect_sample = False

        return sample_noise, sample_no_noise, t 

    def reset(self):
        self._obs = self._env.reset()
        if self._done:
            self.last_n_obs = [np.zeros(params['O']['dim']) for _ in range(params['model']['num_O'])]  
        self._done = False

    def get_observation(self):
        obs_sample = Sample(meta_data=params, T=1)
        im = AgentRCcar.process_image(self._obs)
        obs_sample.set_O(im.ravel(), t=0, sub_obs='camera')
        return obs_sample.get_O(t=0)

    def get_state(self):
        state_sample = Sample(meta_data=params, T=1)
        vel = [self._info['vel']]
        coll = [self._info['coll']]
        state_sample.set_X(vel, t=0, sub_state='velocity')
        state_sample.set_X(vel, t=0, sub_state='collision')
        return state_sample.get_X(t=0)

    def get_coll(self):
        return self._info['coll']

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
        self._obs, _, self._done, self._info = self._env.step(u)
 
    def close(self):
        self._env.close()
