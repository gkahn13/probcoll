
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
from state_utils import quad_to_pitch
from transformations import quaternion_multiply
import matplotlib.pyplot as plt


class Controller():
    def __init__(self,Kp = 0.4,Kd = 0.4, Kp_yaw = 0.5, Kd_yaw = 0.5):
        self.rate = 30.0
        self.terminated = True
    def terminate(self, msg):
        print 'x'
        self.terminated = True

    def start(self, msg):
        print 'start'
        self.terminated = False

    def control(self):
        rospy.init_node('bebop_random_planner', anonymous = True)
        pub_control = rospy.Publisher('/vservo/cmd_vel', Twist, queue_size = 10)
        sub_terminate = rospy.Subscriber('/bebop/collision', Empty, self.terminate)
        sub_start = rospy.Subscriber('/bebop/start_rollout', Empty, self.start)
        # sub_angular_vel = rospy.Subscriber('/ardrome/imu', Imu, self.update_angular_vel)
        r = rospy.Rate(10)
        while not rospy.is_shutdown(): 
                command = Twist()  
                i = 0           
                while not self.terminated:
                    if i % 5 == 0:
                        angle = np.random.uniform(low = -1, high = 1)
                        command.linear.x = 0.8*math.cos(angle)
                        command.linear.y =  0.8*math.sin(angle)
                    pub_control.publish(command)
                    i += 1
                    r.sleep()

if __name__ == '__main__':
    try:
        if len(sys.argv) == 2:
            Kp = float(sys.argv[1])
            controller = Controller(Kp)
        elif len(sys.argv) ==3:
            Kd = float(sys.argv[1])
            Kp = float(sys.argv[2])
            controller = Controller(Kp, Kd)
        else:
            controller = Controller()
        controller.control()
    except rospy.ROSInterruptException: pass