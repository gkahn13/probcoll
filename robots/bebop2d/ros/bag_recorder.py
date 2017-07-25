import rospy
import numpy as np
from std_msgs.msg import Empty
from geometry_msgs.msg import Vector3, Point, Twist, Pose
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import time
import IPython
import sys
import math
import os
from state_utils import quad_to_yaw
import matplotlib.pyplot as plt
import pickle
import rosbag
import rostopic
import threading

class Recorder:
    def __init__(self, topics):
        self.topics = topics
        self.recording = False
        self.cur_bag = None
        self.path = None
        rospy.init_node('recorder', anonymous = True)
        self.sub_terminate = rospy.Subscriber('/bebop/collision', Empty, self.terminate)
        self.sub_start = rospy.Subscriber('/bebop/start_rollout', Empty, self.start)
        self.initialize_subs()
        self.lock = threading.RLock()
    def initialize_subs(self):
        for topic in self.topics:
            msg_type = rostopic.get_topic_class(topic)[0]
            rospy.Subscriber(topic, msg_type, self.subcribe, callback_args=(topic,))
    def start(self, msg):
        if not self.recording:
            path = 'auto_set/'
            self.path = path + str(len(os.listdir(path))) +'.bag'
            self.cur_bag = rosbag.Bag(self.path, 'w')
            self.recording = True
            print 'start recording'
    def terminate(self, msg):
        if self.recording:
            self.recording = False
            self.cur_bag.close()
            self.cur_bag = None
            print 'rollout saved'

    def subcribe(self, msg, args):
        if self.recording:
            topic = args[0]
            with self.lock:
                self.cur_bag.write(topic, msg)

    def record(self):
        r = rospy.Rate(10)
        while not rospy.is_shutdown():  
            r.sleep()
if __name__ == '__main__':
    topics = ['/bebop/cmd_vel',
    '/bebop/collision',
    '/bebop/joint_states',
    '/bebop/odom',
    '/bebop/start_rollout',
    '/bebop/states/ARDrone3/CameraState/Orientation',
    '/bebop/states/ARDrone3/PilotingState/AltitudeChanged',
    '/bebop/states/ARDrone3/PilotingState/AttitudeChanged',
    '/bebop/states/ARDrone3/PilotingState/FlyingStateChanged',
    '/bebop/states/ARDrone3/PilotingState/PositionChanged',
    '/bebop/states/ARDrone3/PilotingState/SpeedChanged',
    '/bebop_vel_ctrl_node/pid_alt/parameter_descriptions',
    '/bebop_vel_ctrl_node/pid_alt/parameter_updates',
    '/bebop_vel_ctrl_node/pid_forward/parameter_descriptions', 
    '/bebop_vel_ctrl_node/pid_forward/parameter_updates', 
    '/bebop_vel_ctrl_node/pid_lateral/parameter_descriptions',
    '/bebop_vel_ctrl_node/pid_lateral/parameter_updates',
    '/bebop_vel_ctrl_node/pid_yaw/parameter_descriptions',
    '/bebop_vel_ctrl_node/pid_yaw/parameter_updates',
    '/vservo/cmd_vel']
    a = Recorder(topics)
    a.record()
