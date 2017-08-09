import numpy as np
import cv2, os
from general.utility.logger import get_logger
import rospy
from std_msgs.msg import Empty
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import geometry_msgs.msg as geometry_msgs
import visualization_msgs.msg as visualization_msgs
import cv_bridge

from general.agent.agent import Agent
import general.ros.ros_utils as ros_utils
from general.state_info.sample import Sample

from config import params

class AgentBebop2d(Agent):

    def __init__(self):
        self.is_teleop = params['planning']['teleop']
        self._save_dir = os.path.join(params['exp_dir'], params['exp_name'])
        self._logger = get_logger(
            self.__class__.__name__,
            params['probcoll']['logger'],
            os.path.join(self._save_dir, 'dagger.txt'))
        self._curr_rollout_t = 0
        # self._ros_start_rollout = ros_utils.RosCallbackEmpty(params['bebop']['topics']['start_rollout'], std_msgs.Empty)
        self._ros_start_rollout = rospy.Subscriber(params['bebop']['topics']['start_rollout'], Empty, self.start)
        self._ros_pub_start = rospy.Publisher(params['bebop']['topics']['start_rollout'], Empty, queue_size=10)
        self._ros_teleop = rospy.Subscriber(params['bebop']['topics']['cmd_vel'], geometry_msgs.Twist, self.receive_teleop)
        self.last_n_obs = [np.zeros(params['O']['dim']) for _ in xrange(params['model']['num_O'])]
        rospy.init_node('DaggerPredictionBebop2d', anonymous=True)
        bebop_topics = params['bebop']['topics']
        self.just_crashed = True
        ### subscribers
        # import IPython; IPython.embed()embed()
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
        self._info = dict()
        self.reset()
        self.cur_teleop_command = None
        # self._info['linearvel'] = np.array([0.0, 0.0, 0.0])

    def start(self, msg):
        self.just_crashed = False
        self._logger.info('Starting')

    def receive_teleop(self, msg):
        self.cur_teleop_command = np.array([msg.linear.x, msg.linear.y, msg.angular.z], dtype='float32')

    def sample_policy(self, policy, T=1, rollout_num=0, is_testing=False, only_noise=False):
        visualize = params['planning'].get('visualize', False)
        sample_noise = Sample(meta_data=params, T=T)
        sample_no_noise = Sample(meta_data=params, T=T)
        r = rospy.Rate(1.0/params['probcoll']['dt'])
        # import IPython; IPython.embed()
        self.just_crashed = False
        while self.is_teleop and self.cur_teleop_command is None:
            r.sleep()
        for t in xrange(T):
            # Get observation and act
            o_t = self.get_observation()
            # import IPython; IPython.embed()
            self.last_n_obs.pop(0)
            self.last_n_obs.append(o_t)
            # TODO only_noise
            # self.coll = o_t[-1]
            if is_testing:
                u_t, u_t_no_noise = policy.act(
                    self.last_n_obs,
                    t,
                    rollout_num,
                    only_noise=only_noise,
                    only_no_noise=is_testing,
                    visualize=visualize)
                if not self.is_teleop:
                    self.act(u_t_no_noise)
                    self._info['linearvel'] = u_t_no_noise
                else:
                    self._info['linearvel'] = self.cur_teleop_command
            else:
                u_t, u_t_no_noise = policy.act(
                    self.last_n_obs,
                    self._curr_rollout_t,
                    rollout_num,
                    only_noise=False,
                    visualize=visualize)
                if not self.is_teleop:
                    print 'noisy act:{0}'.format(u_t)
                    print 'executed act: {0}'.format(u_t_no_noise)
                    self.act(u_t_no_noise)
                    self._info['linearvel'] = u_t
                else:
                    self._info['linearvel'] = self.cur_teleop_command
            x_t = self.get_state()
            r.sleep()
            coll_time = self.coll_callback.get()
            if coll_time is not None:
                self.just_crashed = True
                self.coll = True
                o_t[-1] = 1
            else:
                o_t[-1] = 0
            sample_noise.set_X(x_t, t=t)
            sample_noise.set_O(o_t, t=t)
            sample_no_noise.set_X(x_t, t=t)
            sample_no_noise.set_O(o_t, t=t)
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

    def sample_artificial_policy(self, policy, T=1, rollout_num=0, is_testing=False, only_noise=False):
        visualize = params['planning'].get('visualize', False)
        sample_noise = Sample(meta_data=params, T=T)
        sample_no_noise = Sample(meta_data=params, T=T)
        # o_t_no_collide = 255*np.ones([257], dtype='float32')
        # o_t_collide = np.zeros([257], dtype='float32')
        o_t_collide = np.load('collide.npy')
        o_t_no_collide = np.load('no_collide.npy')
        o_t_collide[-1] = 0
        # o_t_collide[:10] = np.ones([10], dtype='float32')
        # o_t_collide[10] = 1
        # o_t_collide[20] = 1
        # o_t_collide[30] = 1
        o_t_no_collide[-1] = 0
        control1 = np.array([0.6, 0.0, 0.0], dtype='float32')
        control2 = np.array([0.0, 0.0, 0.0], dtype='float32')
        case = np.random.random_integers(0, 1)
        r = rospy.Rate(1.0)
        if case == 0:
            print 'non-collision case'
            for t in xrange(T):
                # Get observation and act
                o_t = o_t_no_collide.copy()
                o_t = o_t + np.random.normal(0, 3, 257)
                o_t = o_t.clip(min=0, max=255)
                o_t[-1] = 0
                if np.random.random_integers(0, 1) == 1:
                    u_t = control1
                else:
                    u_t = control2
                # print u_t
                self.last_n_obs.pop(0)
                self.last_n_obs.append(o_t)
                # TODO only_noise
                self._info['linearvel'] = u_t
                x_t = u_t
                # print x_t
                # Record
                # r.sleep()
                o_t[-1] = 0
                assert o_t[-1] == 0
                sample_noise.set_X(x_t, t=t)
                sample_noise.set_O(o_t, t=t)
                sample_no_noise.set_X(x_t, t=t)
                sample_no_noise.set_O(o_t, t=t)
                if not is_testing:
                    sample_noise.set_U(u_t, t=t)
                # import IPython;IPython.embed()
                if not only_noise:
                    sample_no_noise.set_U(u_t, t=t)
                if self.coll:
                    self._curr_rollout_t = 0
                    break
                else:
                    self._curr_rollout_t += 1
        else:
            print 'collision case'
            for t in xrange(1):
                # Get observation and act
                if t == 0:
                    o_t = o_t_collide.copy()
                    u_t = control1
                else:
                    o_t = o_t_no_collide.copy()
                    if np.random.random_integers(0, 1) == 1:
                        u_t = control1
                    else:
                        u_t = control2
                o_t = o_t + np.random.normal(0, 3, 257)
                o_t = o_t.clip(min=0, max=255)
                o_t[-1] = 0
                self.last_n_obs.pop(0)
                self.last_n_obs.append(o_t)
                # TODO only_noise
                self._info['linearvel'] = u_t
                x_t = u_t
                # print x_t
                # Record
                # r.sleep()
                o_t[-1] = 1
                assert o_t[-1] == 1
                sample_noise.set_X(x_t, t=t)
                sample_noise.set_O(o_t, t=t)
                # sample_noise.set_O([int(self.coll)], t=t, sub_obs='collision')
                # sample_noise.set_O([int(self.coll_callback.get() is not None)], t=t, sub_obs='collision')
                sample_no_noise.set_X(x_t, t=t)
                sample_no_noise.set_O(o_t, t=t)
                # sample_no_noise.set_O([int(self.coll)], t=t, sub_obs='collision')
                # sample_no_noise.set_O([int(self.coll_callback.get() is not None)], t=t, sub_obs='collision')
                if not is_testing:
                    sample_noise.set_U(u_t, t=t)
                # import IPython;IPython.embed()
                if not only_noise:
                    sample_no_noise.set_U(u_t, t=t)
                if t == T -1:
                    self._curr_rollout_t = 0
                    break
                else:
                    self._curr_rollout_t += 1

            # import IPython; IPython.embed()
        return sample_noise, sample_no_noise, t

    def reset(self, pos=None, ori=None, hard_reset=False):
        # self._obs = self.env.reset(pos=pos, hpr=ori, hard_reset=hard_reset)
        self.act(None)  # stop bebop
        self.cur_teleop_command = None
        # if self.just_crashed:
        self._logger.info('Press start')
        # while self.just_crashed and not rospy.is_shutdown():
        self.just_crashed = True
        while self.just_crashed and not rospy.is_shutdown():
            rospy.sleep(0.1)
        self.coll_callback.get()
        self.coll = False
        if hard_reset:
            self.last_n_obs = [np.zeros(params['O']['dim']) for _ in xrange(params['model']['num_O'])]

    def get_state(self):
        state_sample = Sample(meta_data=params, T=1)
        vel = self._info['linearvel']
        state_sample.set_X(vel, t=0, sub_state='linearvel')
        return state_sample.get_X(t=0)

    def get_observation(self):
        obs_sample = Sample(meta_data=params, T=1)
        ### collision
        # is_coll = coll_time is not None
        obs_sample.set_O([0], t=0, sub_obs='collision')

        ### camera
        image_msgs = self.image_callback.get()
        assert(len(image_msgs) > 0)
        ### keep only those before the collision
        # if self.coll:
        #     image_msgs_filt = [im for im in image_msgs if im.header.stamp.to_sec() < coll_time.to_sec()]
        #     if len(image_msgs_filt) == 0:
        #         image_msgs_filt = [image_msgs[-1]]
        #     image_msgs = image_msgs_filt
        image_msg = image_msgs[-1]
        im = AgentBebop2d.process_image(image_msg, self.cv_bridge)
        # im = np.zeros((params['O']['camera']['height'], params['O']['camera']['width']), dtype=np.float32)
        obs_sample.set_O(im.ravel(), t=0, sub_obs='camera')
        ### publish image for debugging
        # self.im_debug_pub.publish(self.cv_bridge.cv2_to_imgmsg((255. * im).astype(np.uint8), 'mono8'))
        return obs_sample.get_O(t=0)

    @staticmethod
    def process_image(image_msg, cvb):
        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

        image = rgb2gray(cvb.imgmsg_to_cv2(image_msg).astype(np.float32))
        im = cv2.resize(image,
                                    (params['O']['camera']['height'], params['O']['camera']['width']),
                                    interpolation=cv2.INTER_AREA) # TODO how does this deal with aspect ratio

        return im

    def act(self, u):
        if u is not None:
            print u
            if u[0] == 0 and u[1]== 0:
                twist_msg = geometry_msgs.Twist(
                    linear=geometry_msgs.Point(0., 0., 0.),
                    angular=geometry_msgs.Point(0., 0., 0)
                )
                self.cmd_vel_pub.publish(None)
                # r.sleep()
                # self.cmd_vel_pub.publish(twist_msg)
                # r.sleep()
                # self.cmd_vel_pub.publish(twist_msg)
                # r.sleep()
                # self.cmd_vel_pub.publish(twist_msg)
                # r.sleep()
                # self.cmd_vel_pub.publish(None)
                # r.sleep()
                # self.cmd_vel_pub.publish(None)
                # self._ros_pub_start.publish(Empty())

            else:
                twist_msg = geometry_msgs.Twist(
                    linear=geometry_msgs.Point(u[0], u[1], 0.),
                    angular=geometry_msgs.Point(0., 0., 0)
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

    def close(self):
        pass
