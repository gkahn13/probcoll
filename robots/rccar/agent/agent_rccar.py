import numpy as np
import cv2
import rospy
import sensor_msgs
import std_msgs
import geometry_msgs
import cv_bridge
import os
import time

try:
    import bair_car.srv
except:
    pass

from general.utility.logger import get_logger
from general.agent.agent import Agent
from config import params
import robots.rccar.ros.ros_utils as ros_utils
from general.state_info.sample import Sample

class AgentRCcar(Agent):

    def __init__(self, dynamics):
        Agent.__init__(self, dynamics)
        rccar_topics = params['rccar']['topics']
        self._logger = get_logger(
            self.__class__.__name__,
            'info',
            os.path.join(
                os.path.join(params['exp_dir'], params['exp_name']),
                'debug.txt'))

        self.sim = params['world']['sim']

        if self.sim:
            service = params['sim']['srv']
            self._logger.info("Waiting for service")
            ros_utils.wait_for_service(service)
            self._logger.info("Connected to service")
            self.srv = ros_utils.ServiceProxy(service, bair_car.srv.sim_env)
            data =  self.srv(reset=True)
            self.sim_coll = data.coll
            self.sim_image = data.image
            self.sim_depth = data.depth
            self.sim_back_image = data.back_image
            self.sim_back_depth = data.back_depth
            self.sim_state = data.pose
            self.sim_vel = 0.0
            self.sim_steer = 0.0
            self.sim_reset = False
            self.sim_last_coll = False

        self.last_n_obs = [np.zeros(params['O']['dim']) for _ in xrange(params['model']['num_O'])]  

        ### subscribers
        self.image_callback = ros_utils.RosCallbackAll(rccar_topics['camera'], sensor_msgs.msg.Image,
                                                       max_num_msgs=2, clear_msgs=False)
        self.coll_callback = ros_utils.RosCallbackEmpty(rccar_topics['collision'], std_msgs.msg.Empty)

        ### publishers
        self.cmd_steer_pub = ros_utils.Publisher(rccar_topics['cmd_steer'], std_msgs.msg.Float32, queue_size=10)
        self.cmd_vel_pub = ros_utils.Publisher(rccar_topics['cmd_vel'], std_msgs.msg.Float32, queue_size=10)
        self.pred_image_pub = ros_utils.Publisher(rccar_topics['pred_image'], sensor_msgs.msg.Image, queue_size=1)

        self.cv_bridge = cv_bridge.CvBridge()

    def sample_policy(self, x0, policy, rollout_num, T=None, use_noise=True, only_noise=False, **policy_args):
        if T is None:
            T = policy._T
        rate = rospy.Rate(1. / params['dt'])
        policy_sample = Sample(meta_data=params, T=T)
        policy_sample.set_X(x0, t=0)
        policy_sample_no_noise = Sample(meta_data=params, T=T)
        visualize = params['planning'].get('visualize', False)
        for t in xrange(T):
            # get observation and act
            x_t = policy_sample.get_X(t=t)
            o_t = self.get_observation(x_t)
            self.last_n_obs.pop()
            self.last_n_obs.insert(0, o_t)
            if self.sim:
                x_t = self._get_sim_state(x_t) 
            start = time.time()
            u_t, u_t_no_noise = policy.act(self.last_n_obs, t, rollout_num, only_noise=only_noise, visualize=visualize)
            self._logger.debug(time.time() - start)
            # only execute control if no collision
            # TODO is this necessary
            if int(o_t[policy_sample.get_O_idxs(sub_obs='collision')][0]) == 0:
                if use_noise:
                    self.execute_control(u_t)
                else:
                    self.execute_control(u_t_no_noise)

            # record
            policy_sample.set_X(x_t, t=t)
            policy_sample.set_O(o_t, t=t)
            policy_sample.set_U(u_t, t=t)
            if not only_noise:
                policy_sample_no_noise.set_U(u_t_no_noise, t=t)
            
            # In sim we do not have cycles
            if self.sim:
                policy_sample.set_O([int(self.sim_coll)], t=t, sub_obs='collision')
            else:
                # propagate dynamics
                if t < T-1:
                    x_tp1 = self._dynamics.evolve(x_t, u_t)
                    policy_sample.set_X(x_tp1, t=t+1)
           
                rate.sleep()
                # see if collision in the past cycle
                policy_sample.set_O([int(self.coll_callback.get() is not None)], t=t, sub_obs='collision')

        return policy_sample, policy_sample_no_noise

    def reset(self, x):
        pass

    def _get_sim_state(self, xt):
        state_sample = Sample(meta_data=params, T=1)
        state_msg = self.sim_state
        state = np.array([
                state_msg.position.x,
                state_msg.position.y,
                state_msg.position.z,
                state_msg.orientation.x,
                state_msg.orientation.y,
                state_msg.orientation.z,
                state_msg.orientation.w
            ])
        state_sample.set_X(state[:3], t=0, sub_state='position')
        state_sample.set_X(state[3:], t=0, sub_state='orientation')
        x = np.nan_to_num(xt) + np.nan_to_num(state_sample.get_X(t=0))
        return x

    def get_sim_state_data(self):
        state_msg = self.sim_state
        pos = [
                state_msg.position.x,
                state_msg.position.y,
                state_msg.position.z,
            ]
        quat = [
                state_msg.orientation.x,
                state_msg.orientation.y,
                state_msg.orientation.z,
                state_msg.orientation.w
            ]
        return pos, quat

    def get_observation(self, x):
        obs_sample = Sample(meta_data=params, T=1)

        if self.sim:
            is_coll = self.sim_coll
            image_msg = self.sim_image
            depth_msg = self.sim_depth
            back_image_msg = self.sim_back_image
            back_depth_msg = self.sim_back_depth
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
        
        if params['O'].get('use_depth', False):
            im = AgentRCcar.process_depth(depth_msg, self.cv_bridge)
            back_im = AgentRCcar.process_depth(back_depth_msg, self.cv_bridge)
        else:
            im = AgentRCcar.process_image(image_msg, self.cv_bridge)
            back_im = AgentRCcar.process_image(back_image_msg, self.cv_bridge)

        ros_image = self.cv_bridge.cv2_to_imgmsg(im, "mono8")
        self.pred_image_pub.publish(ros_image)
        obs_sample.set_O(im.ravel(), t=0, sub_obs='camera')
        obs_sample.set_O(back_im.ravel(), t=0, sub_obs='back_camera')
        obs_sample.set_O([int(is_coll)], t=0, sub_obs='collision')
        return obs_sample.get_O(t=0)

    @staticmethod
    def process_depth(depth_msg, cvb):
        image = (cvb.imgmsg_to_cv2(depth_msg))
        mono_image = np.array(np.fromstring(image.tostring(), np.int32), np.float32)
        # TODO this is hardcoded
        mono_image = (1.0653532e9 - mono_image) / (1.76e5) * 255 
        im = cv2.resize(
            np.reshape(mono_image, (image.shape[0], image.shape[1])),
            (params['O']['camera']['height'], params['O']['camera']['width']),
            interpolation=cv2.INTER_AREA)
        return im.astype(np.uint8)

    @staticmethod
    def process_image(image_msg, cvb):
        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

        image = rgb2gray(cvb.imgmsg_to_cv2(image_msg)).astype(np.uint8)
        im = cv2.resize(
            image,
            (params['O']['camera']['height'], params['O']['camera']['width']),
            interpolation=cv2.INTER_AREA) #TODO how does this deal with aspect ratio 
        return im

    def execute_control(self, u, reset=False, pos=None, quat=None):
        if u is not None:
            s = Sample(meta_data=params, T=2)
            steer = u[s.get_U_idxs('cmd_steer')][0]
            vel = u[s.get_U_idxs('cmd_vel')][0]
            self.cmd_steer_pub.publish(std_msgs.msg.Float32(steer))
            self.cmd_vel_pub.publish(std_msgs.msg.Float32(vel))
            if self.sim:
                data= self.srv(steer=steer, vel=vel, reset=reset)
        else:
            self.cmd_steer_pub.publish(std_msgs.msg.Float32(49.5))
            self.cmd_vel_pub.publish(std_msgs.msg.Float32(0.))
            if self.sim:
                if reset:
                    if quat is None:
                        quat = [0.0, 0.0, 0.0, 0.0]
                    if pos is None:
                        if len(params['world']['testing']['positions']) > 0:
                            pos = params['world']['testing']['positions'][np.random.randint(len(params['world']['testing']['positions']))]
                        else:
                            pos = [0.0, 0.0, 0.0]
                    pose = geometry_msgs.msg.Pose()
                    pose.position.x, pose.position.y, pose.position.z = pos
                    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
                    data = self.srv(reset=True, pose=pose)
                else:
                    data = self.srv()
        if self.sim:
            self.sim_last_coll = self.sim_coll
            self.sim_coll = data.coll
            self.sim_image = data.image
            self.sim_depth = data.depth
            self.sim_back_image = data.back_image
            self.sim_back_depth = data.back_depth
            self.sim_state = data.pose
