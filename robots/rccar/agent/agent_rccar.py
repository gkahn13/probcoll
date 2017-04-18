import numpy as np
import cv2

import rospy
import sensor_msgs
import std_msgs
import cv_bridge

try:
    import bair_car.srv
except:
    pass

from general.agent.agent import Agent
from config import params
import general.ros.ros_utils as ros_utils

from general.state_info.sample import Sample
from general.policy.noise_models import ZeroNoise

class AgentRCcar(Agent):

    def __init__(self, dynamics):
        Agent.__init__(self, dynamics)
        rccar_topics = params['rccar']['topics']
        self.sim = params['rccar']['sim']
       
        if self.sim:
            service = params['rccar']['srv']
            rospy.wait_for_service(service)
            self.srv = rospy.ServiceProxy(service, bair_car.srv.sim_env)
            # TODO use depth
            data =  self.srv(reset=True)
            self.sim_coll, self.sim_image = data.coll, data.image
            self.sim_vel = 0.0
            self.sim_steer = 0.0
            self.sim_reset = False

        ### subscribers
        self.image_callback = ros_utils.RosCallbackAll(rccar_topics['camera'], sensor_msgs.msg.Image,
                                                       max_num_msgs=2, clear_msgs=False)
        self.coll_callback = ros_utils.RosCallbackEmpty(rccar_topics['collision'], std_msgs.msg.Empty)

        ### publishers
        self.cmd_steer_pub = rospy.Publisher(rccar_topics['cmd_steer'], std_msgs.msg.Float32, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher(rccar_topics['cmd_vel'], std_msgs.msg.Float32, queue_size=10)
        self.pred_image_pub = rospy.Publisher(rccar_topics['pred_image'], sensor_msgs.msg.Image, queue_size=1)

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
                x_tp1 = self._dynamics.evolve(x_t, u_t)
                policy_sample.set_X(x_tp1, t=t+1)
           
            # In sim we do not have cycles
            if self.sim:
                policy_sample.set_O([int(self.sim_coll)], t=t, sub_obs='collision')
            else:
                rate.sleep()
                # see if collision in the past cycle
                policy_sample.set_O([int(self.coll_callback.get() is not None)], t=t, sub_obs='collision')

        return policy_sample

    def reset(self, x):
        pass

    def get_observation(self, x):
        obs_sample = Sample(meta_data=params, T=2)

        if self.sim:
            is_coll = self.sim_coll
            image_msg = self.sim_image
        else:
            ### collision
            coll_time = self.coll_callback.get()
            is_coll = coll_time is not None

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
	ros_image = self.cv_bridge.cv2_to_imgmsg((im*255).astype(np.uint8), "mono8")
        self.pred_image_pub.publish(ros_image)
        # im = np.zeros((params['O']['camera']['height'], params['O']['camera']['width']), dtype=np.float32)
        obs_sample.set_O(im.ravel(), t=0, sub_obs='camera')
        obs_sample.set_O([int(is_coll)], t=0, sub_obs='collision')
        
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
            steer = u[s.get_U_idxs('cmd_steer')][0]
            vel = u[s.get_U_idxs('cmd_vel')][0]
            self.cmd_steer_pub.publish(std_msgs.msg.Float32(steer))
            self.cmd_vel_pub.publish(std_msgs.msg.Float32(vel))
            if self.sim:
                data= self.srv(steer=steer, vel=vel)
        else:
            self.cmd_steer_pub.publish(std_msgs.msg.Float32(49.5))
            self.cmd_vel_pub.publish(std_msgs.msg.Float32(0.))
            if self.sim:
                data = self.srv(reset=True)
        if self.sim:
            self.sim_coll, self.sim_image = data.coll, data.image
