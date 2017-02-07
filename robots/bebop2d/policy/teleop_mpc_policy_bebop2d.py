import numpy as np

import rospy
import geometry_msgs.msg as geometry_msgs
import bebop_msgs.msg as bebop_msgs

import general.ros.ros_utils as ros_utils

from rll_quadrotor.policy.policy import Policy
from rll_quadrotor.utility.logger import get_logger

class TeleopMPCPolicyBebop2d(Policy):

    def __init__(self, meta_data):
        use_obs = False
        Policy.__init__(self, meta_data['mpc']['H'], use_obs, meta_data)
        self.logger = get_logger(self.__class__.__name__, 'warn')

        self.cmd_vel_callback = ros_utils.RosCallbackAll(meta_data['bebop']['topics']['cmd_vel'],
                                                         geometry_msgs.Twist,
                                                         max_num_msgs=2) # so that don't just get the cmd_vel we sent
        self.measured_vel_callback = ros_utils.RosCallbackMostRecent(meta_data['bebop']['topics']['measured_vel'],
                                                                     bebop_msgs.Ardrone3PilotingStateSpeedChanged)

        while not rospy.is_shutdown():
            measured_vel = self.measured_vel_callback.get()
            if measured_vel is not None:
                self.last_measured_vel = measured_vel
                break

    def act(self, x, obs, t, noise, ref_traj=None):
        vel_msgs = self.cmd_vel_callback.get()
        if len(vel_msgs) < 2:
            self.logger.warn('Using measured_vel')
            measured_vel = self.measured_vel_callback.get()
            if measured_vel is not None:
                self.last_measured_vel = measured_vel
            u = np.array([self.last_measured_vel.speedX, self.last_measured_vel.speedY])
        else:
            vel_msg = vel_msgs[-1]
            u = np.array([vel_msg.linear.x, vel_msg.linear.y])

        # measured_vel = self.measured_vel_callback.get()
        # if measured_vel is not None:
        #     self.last_measured_vel = measured_vel
        # u = np.array([self.last_measured_vel.speedX, self.last_measured_vel.speedY])

        return u

    def get_info(self):
        return dict()
