import numpy as np
import cv2

import rospy
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import cv_bridge

from general.agent.agent import Agent

import general.ros.ros_utils as ros_utils

from rll_quadrotor.state_info.sample import Sample
from rll_quadrotor.policy.noise_models import ZeroNoise

from config import params

class AgentRCcar(Agent):

    def __init__(self, dynamics):
        Agent.__init__(self, dynamics)
        rccar_topics = params['rccar']['topics']

        ### subscribers
        self.image_callback = ros_utils.RosCallbackAll(rccar_topics['camera'], sensor_msgs.Image,
                                                       max_num_msgs=2, clear_msgs=False)
        self.coll_callback = ros_utils.RosCallbackEmpty(rccar_topics['collision'], std_msgs.Empty)

        ### publishers
        self.cmd_steer_pub = rospy.Publisher(rccar_topics['cmd_steer'], std_msgs.Float32, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher(rccar_topics['cmd_vel'], std_msgs.Float32, queue_size=10)

        self.cv_bridge = cv_bridge.CvBridge()

    def sample_policy(self, x0, policy, T=None, **policy_args):
        if T is None:
            T = policy._T
        policy_sample = Sample(meta_data=params, T=T)
        noise = policy_args.get('noise', ZeroNoise(params))

        rate = rospy.Rate(1. / params['dt'])
        policy_sample.set_X(x0, t=0)
        for t in xrange(T):
            # get observation and act
            x_t = policy_sample.get_X(t=t)
            o_t = self.get_observation(x_t)
            import time
            start = time.time()
            u_t = policy.act(x_t, o_t, t, noise=noise)
            print time.time() - start
            # only execute control if no collision
            if int(o_t[policy_sample.get_O_idxs(sub_obs='collision')][0]) == 0:
                self.execute_control(u_t)

            # record
            policy_sample.set_X(x_t, t=t)
            policy_sample.set_O(o_t, t=t)
            policy_sample.set_U(u_t, t=t)

            # propagate dynamics
            if t < T-1:
                x_tp1 = self.dynamics.evolve(x_t, u_t)
                policy_sample.set_X(x_tp1, t=t+1)
            rate.sleep()

            # see if collision in the past cycle
            policy_sample.set_O([int(self.coll_callback.get() is not None)], t=t, sub_obs='collision')

        return policy_sample

    def reset(self, x):
        pass

    def get_observation(self, x):
        obs_sample = Sample(meta_data=params, T=2)

        ### collision
        coll_time = self.coll_callback.get()
        is_coll = coll_time is not None
        obs_sample.set_O([int(is_coll)], t=0, sub_obs='collision')

        ### camera
        image_msgs = self.image_callback.get()
        assert(len(image_msgs) > 0)
        ### keep only those before the collision
        if is_coll:
            image_msgs_filt = [im for im in image_msgs if im.header.stamp.to_sec() < coll_time.to_sec()]
            if len(image_msgs_filt) == 0:
                image_msgs_filt = [image_msgs[-1]]
            image_msgs = image_msgs_filt
        image_msg = image_msgs[-1]
        im = AgentRCcar.process_image(image_msg, self.cv_bridge)
        # im = np.zeros((params['O']['camera']['height'], params['O']['camera']['width']), dtype=np.float32)
        obs_sample.set_O(im.ravel(), t=0, sub_obs='camera')

        return obs_sample.get_O(t=0)

    @staticmethod
    def process_image(image_msg, cvb):
        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

        image = rgb2gray(cvb.imgmsg_to_cv2(image_msg).astype(np.float32))
        im = (1./255.) * cv2.resize(image,
                                    (params['O']['camera']['height'], params['O']['camera']['width']),
                                    interpolation=cv2.INTER_AREA) # TODO how does this deal with aspect ratio

        return im

    def execute_control(self, u):
        if u is not None:
            s = Sample(meta_data=params, T=2)
            self.cmd_steer_pub.publish(std_msgs.Float32(u[s.get_U_idxs('cmd_steer')][0]))
            self.cmd_vel_pub.publish(std_msgs.Float32(u[s.get_U_idxs('cmd_vel')][0]))
        else:
            self.cmd_steer_pub.publish(std_msgs.Float32(50.))
            self.cmd_vel_pub.publish(std_msgs.Float32(0.))
