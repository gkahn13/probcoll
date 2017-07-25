
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
from state_utils import quad_to_yaw
import matplotlib.pyplot as plt
from transformations import euler_from_quaternion

class Detector():
    def __init__(self):
        self.rate = 30.0
        self.terminated = False
        self.completed = False
        self.pub_crash =rospy.Publisher('/bebop/collision', Empty, queue_size=10)
        self.pub_start = rospy.Publisher('/bebop/start_rollout', Empty, queue_size = 10)
        self.pub_control = rospy.Publisher('/vservo/cmd_vel', Twist, queue_size = 10)
        back_up_command = Twist()
        back_up_command.linear.x = -0.75
        left_rotate_command = Twist()
        left_rotate_command.angular.z = 1
        self.recovery_commands = [Twist()]*5 + [back_up_command]*10 + [Twist()]*5 + [left_rotate_command]*15 + [Twist()]*10
        self.recovery_index = 0
        self.start_recovery = False
        self.last_control = None
        self.prev_states = []
    def terminate(self, msg):
        print 'x'
        self.terminated = True
        self.start_recovery = True

    def land(self, msg):
        print 'land'
        self.completed = True

    def takeoff(self, msg):
        print 'takeoff'
        self.completed = False

    def start(self, msg):
        print 'start'
        self.terminated = False
        self.sequences = []
        self.velocities = []
    
    def detect_crash(self):
        if len(self.prev_states) < 6:
            return False
        cur_state = self.prev_states[-1]
        last_state = self.prev_states[-2]
        second_last_state = self.prev_states[-3]
        t_linear_x, t_linear_y, t_des_linear_x, t_des_linear_y, t_pos_z, t_angular_x, t_angular_y, t_angular_z = cur_state
        t_b1_linear_x, t_b1_linear_y, t_b1_des_linear_x, t_b1_des_linear_y, t_b1_pos_z, t_b1_angular_x, t_b1_angular_y, t_b1_angular_z = last_state
        t_b2_linear_x, t_b2_linear_y, t_b2_des_linear_x, t_b2_des_linear_y, t_b2_pos_z, t_b2_angular_x, t_b2_angular_y, t_b2_angular_z = second_last_state
        t_b3_linear_x, t_b3_linear_y, t_b3_des_linear_x, t_b3_des_linear_y, t_b3_pos_z, t_b3_angular_x, t_b3_angular_y, t_b3_angular_z = self.prev_states[-4]
        t_b4_linear_x, t_b4_linear_y, t_b4_des_linear_x, t_b4_des_linear_y, t_b4_pos_z, t_b4_angular_x, t_b4_angular_y, t_b4_angular_z = self.prev_states[-5]
        t_b5_linear_x, t_b5_linear_y, t_b5_des_linear_x, t_b5_des_linear_y, t_b5_pos_z, t_b5_angular_x, t_b5_angular_y, t_b5_angular_z = self.prev_states[-6]
        if t_linear_x - t_b1_linear_x < -0.35:
            return True

        delta_c_x = t_b2_des_linear_x - t_linear_x 
        delta_c_y =  abs(t_b2_des_linear_y - t_linear_y)
        delta_l_x = t_b3_des_linear_x - t_b1_linear_x 
        delta_l_y = abs(t_b3_des_linear_y - t_b1_linear_y)
        delta_sl_x = t_b4_des_linear_x - t_b2_linear_x 
        delta_sl_y = abs(t_b4_des_linear_y - t_b2_linear_y)
        if t_b2_des_linear_y == t_b4_des_linear_y and t_b4_des_linear_y == t_b3_des_linear_y and delta_c_y > delta_l_y and delta_l_y > delta_sl_y and delta_c_y > 0.35:
            return True
        return False

    def update_vel(self, msg):
        if self.last_control:
            quat = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, 
            msg.pose.pose.orientation.y, msg.pose.pose.orientation.z]
            x, y, z = x, y, z = euler_from_quaternion(quat)
            state = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, self.last_control.linear.x,
            self.last_control.linear.y, msg.pose.pose.position.z, x%math.pi, y%math.pi, z%math.pi]
            self.prev_states.append(state)
        if self.detect_crash():
            self.pub_crash.publish(Empty())
            self.start_recovery = True

    def evolve_recovery(self):
        if self.recovery_index < len(self.recovery_commands):
            self.pub_control.publish(self.recovery_commands[self.recovery_index])
            self.recovery_index += 1
        else:
            self.recovery_index = 0
            self.just_crashed = False
            print "recovery completed"
            self.start_recovery = False
            self.prev_states = []
            # r = rospy.Rate(20)
            self.pub_start.publish(Empty())
            # r.sleep()
            # self.pub_start.publish(Empty())



    def update_control(self, msg):
        self.last_control = msg

    def control(self):
        rospy.init_node('bebop_crash_detector', anonymous = True)
        pub_control = rospy.Publisher('/vservo/cmd_vel', Twist, queue_size = 10)
        sub_terminate = rospy.Subscriber('/bebop/collision', Empty, self.terminate)
        sub_start = rospy.Subscriber('/bebop/start_rollout', Empty, self.start)
        sub_land = rospy.Subscriber('/bebop/land', Empty, self.land)
        sub_takeoff = rospy.Subscriber('/bebop/takeoff', Empty, self.takeoff)
        pub_crash = rospy.Publisher('/bebop/collision', Empty, queue_size=10)
        sub_control = rospy.Subscriber('/vservo/cmd_vel', Twist, self.update_control)
        sub_pos = rospy.Subscriber('/bebop/odom', Odometry, self.update_vel)
        # sub_angular_vel = rospy.Subscriber('/ardrome/imu', Imu, self.update_angular_vel)
        r = rospy.Rate(10)

        while not rospy.is_shutdown():                
            ys = []
            i = 0
            while not self.completed:
                while not self.terminated:
                    r.sleep()
                while self.start_recovery:
                    self.evolve_recovery()
                    r.sleep()

if __name__ == '__main__':
    try:
        detector = Detector()
        detector.control()
    except rospy.ROSInterruptException: pass