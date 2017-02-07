import numpy as np
import cv2

import rospy
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import geometry_msgs.msg as geometry_msgs
import visualization_msgs.msg as visualization_msgs
import cv_bridge

from general.agent.agent import Agent

import general.ros.ros_utils as ros_utils

from rll_quadrotor.state_info.sample import Sample
from rll_quadrotor.policy.noise_models import ZeroNoise

from config import params

class AgentBebop2d(Agent):

    def __init__(self, dynamics):
        Agent.__init__(self, dynamics)
        bebop_topics = params['bebop']['topics']

        ### subscribers
        self.image_callback = ros_utils.RosCallbackAll(bebop_topics['image'], sensor_msgs.Image,
                                                       max_num_msgs=1)
        self.coll_callback = ros_utils.RosCallbackEmpty(bebop_topics['collision'], std_msgs.Empty)

        ### publishers
        self.cmd_vel_pub = ros_utils.RatePublisher(10., bebop_topics['cmd_vel'], geometry_msgs.Twist)

        ### debugging
        self.im_debug_pub = rospy.Publisher(bebop_topics['debug_image'], sensor_msgs.Image)
        self.cmd_vel_debug_pub = rospy.Publisher(bebop_topics['debug_cmd_vel'],
                                                 visualization_msgs.Marker, queue_size=100)

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
            u_t = policy.act(x_t, o_t, t, noise=noise)
            if params['prediction']['dagger']['planner_type'] != 'teleop': # TODO hack
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
        im = AgentBebop2d.process_image(image_msg, self.cv_bridge)
        # im = np.zeros((params['O']['camera']['height'], params['O']['camera']['width']), dtype=np.float32)
        obs_sample.set_O(im.ravel(), t=0, sub_obs='camera')
        ### publish image for debugging
        self.im_debug_pub.publish(self.cv_bridge.cv2_to_imgmsg((255. * im).astype(np.uint8), 'mono8'))

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
            twist_msg = geometry_msgs.Twist(
                linear=geometry_msgs.Point(u[0], u[1], 0.),
                angular=geometry_msgs.Point(0., 0., 0.)
            )
            self.cmd_vel_pub.publish(twist_msg)

            ### publish marker for debugging
            marker = visualization_msgs.Marker()
            marker.header.frame_id = '/map'
            marker.type = marker.ARROW
            marker.action = marker.ADD
            marker.points = [
                geometry_msgs.Point(0, 0, 0),
                geometry_msgs.Point(u[0], u[1], 0)
            ]
            marker.lifetime = rospy.Duration()
            marker.scale.x = 0.05
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 1.0
            self.cmd_vel_debug_pub.publish(marker)
        else:
            self.cmd_vel_pub.publish(None)