import numpy as np
import cv2
import os
from general.agent.agent import Agent
from general.state_info.sample import Sample
from robots.sim_rccar.simulation.square_env import SquareEnv

from config import params

class AgentSimRCcar(Agent):

    def __init__(self):
        if params['world']['sim']['sim_env'] == 'square':
            self.env = SquareEnv(params['world']['sim'])
        else:
            raise NotImplementedError(
                "Environment {0} is not valid".format(
                    params['world']['sim']['sim_env']))

        self.obs = self.env.reset()
        self.last_n_obs = [np.zeros(params['O']['dim']) for _ in xrange(params['model']['num_O'])]  

    def sample_policy(self, policy, rollout_num, T=None, time_step=0, use_noise=True, only_noise=False, **policy_args):
        if T is None:
            T = policy._T
        policy_sample = Sample(meta_data=params, T=T)
        policy_sample_no_noise = Sample(meta_data=params, T=T)
        visualize = params['planning'].get('visualize', False)
        for t in xrange(T):
            # Get observation and act
            x_t = self.get_state()
            o_t = self.get_observation()
            self.last_n_obs.pop(0)
            self.last_n_obs.append(o_t)
            u_t, u_t_no_noise = policy.act(self.last_n_obs, time_step + t, rollout_num, only_noise=only_noise, visualize=visualize)
            # Only execute control if no collision
            if not self.coll:
                if use_noise:
                    self.act(u_t)
                else:
                    self.act(u_t_no_noise)

            # Record
            policy_sample.set_X(x_t, t=t)
            policy_sample.set_O(o_t, t=t)
            policy_sample.set_U(u_t, t=t)
            if not only_noise:
                policy_sample_no_noise.set_U(u_t_no_noise, t=t)
            
            # In sim we do not have cycles
            policy_sample.set_O([int(self.coll)], t=t, sub_obs='collision')

        return policy_sample, policy_sample_no_noise

    def reset(self, pos=None, ori=None):
        if pos is None:
            if params['world']['testing'].get('position_ranges', None) is not None:
                ranges = params['world']['testing']['position_ranges'] 
                ran = ranges[np.random.randint(len(ranges))]
                pos_ori = np.random.uniform(ran[0], ran[1])
                pos = pos_ori[:3]
                ori = np.array([0.0, 0.0, pos_ori[3]])
            elif len(params['world']['testing']['positions']) > 0:
                pos_ori = params['world']['testing']['positions'][np.random.randint(len(params['world']['testing']['positions']))]
                pos = pos_ori[:3]
                ori = np.array([0.0, 0.0, pos_ori[3]])
        self.obs = self.env.reset(pos=pos, hpr=ori)
        self.last_n_obs = [np.zeros(params['O']['dim']) for _ in xrange(params['model']['num_O'])]  

    def get_observation(self):
        obs_sample = Sample(meta_data=params, T=1)
        front_image = self.obs['front_image']
        front_depth = self.obs['front_depth']
        back_image = self.obs['back_image']
        back_depth = self.obs['back_depth']
        vel = self.obs['vel']
        if params['O'].get('use_depth', False):
            im = AgentSimRCcar.process_depth(front_depth)
            back_im = AgentSimRCcar.process_depth(back_depth)
        else:
            im = AgentSimRCcar.process_image(front_image)
            back_im = AgentSimRCcar.process_image(back_image)

        obs_sample.set_O(im.ravel(), t=0, sub_obs='camera')
        obs_sample.set_O(back_im.ravel(), t=0, sub_obs='back_camera')
        obs_sample.set_O([vel], t=0, sub_obs='vel')
        return obs_sample.get_O(t=0)

    def get_state(self):
        state_sample = Sample(meta_data=params, T=1)
        pos = self.obs['pos']
        ori = self.obs['hpr']
        state_sample.set_X(pos, t=0, sub_state='position')
        state_sample.set_X(ori, t=0, sub_state='orientation')
        return state_sample.get_X(t=0)

    def get_pos_ori(self):
        return self.obs['pos'], self.obs['hpr']

    @property
    def coll(self):
        return self.obs['coll']

    @staticmethod
    def process_depth(image):
        mono_image = np.array(np.fromstring(image.tostring(), np.int32), np.float32)
        # TODO this is hardcoded
        mono_image = (1.0653532e9 - mono_image) / (1.76e5) * 255 
        im = cv2.resize(
            np.reshape(mono_image, (image.shape[0], image.shape[1])),
            (params['O']['camera']['height'], params['O']['camera']['width']),
            interpolation=cv2.INTER_AREA)
        return im.astype(np.uint8)

    @staticmethod
    def process_image(image):
        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

        image = rgb2gray(image).astype(np.uint8)
        im = cv2.resize(
            image,
            (params['O']['camera']['height'], params['O']['camera']['width']),
            interpolation=cv2.INTER_AREA) #TODO how does this deal with aspect ratio 
        return im

    def act(self, u=[0.0, 0.0]):
        self.obs, _, _, _ = self.env.step(u)
    
    def close(self):
        pass
