
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
import pickle
import os
import rosbag

def save(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class Controller():
    def __init__(self,Kp = 0.4,Kd = 0.4, Kp_yaw = 0.5, Kd_yaw = 0.5):
        self.rate = 30.0
        self.terminated = False
        self.completed = False
        self.Kp = Kp
        self.Kd = Kd
        self.Kp_yaw = Kp_yaw
        self.Kd_yaw = Kd_yaw
        self.last_pos = Vector3(0, 0, 0)
        self.last_pos_time = -1
        self.last_pos_seq = -1
        self.last_angular_vel = Vector3(0, 0, 0)
        self.last_angular_vel_time = -1
        self.desired_vel = Vector3()
        self.cur_vel = Vector3()
        self.cur_angular_z = 0
        self.prev_vel = Vector3()
        self.prev_angular_vel = Vector3()
        self.sequences = []
        self.velocities = []
        self.odoms = []
        self.just_crashed = False
        self.cur_control = Twist()
        self.pub_crash =rospy.Publisher('/bebop/collision', Empty, queue_size=10)
        self.pub_start = rospy.Publisher('/bebop/start_rollout', Empty, queue_size = 10)
        back_up_command = Twist()
        back_up_command.linear.x = -0.75
        left_rotate_command = Twist()
        left_rotate_command.angular.z = 1
        self.recovery_commands = [Twist()]*20 + [back_up_command]*15 + [Twist()]*5 + [left_rotate_command]*15 + [Twist()]*10
        self.recovery_index = 0
        self.start_recovery = False
        self.des_control = Twist()
        self.updated_vel = False
        self.init_collision_time = -1
        self.saved = True

    def terminate(self, msg):
        print 'x'
        self.terminated = True
        self.start_recovery = True
        if not self.saved and len(self.odoms) > 10:
            # temp_dict = dict()
            # temp_dict['odoms'] = self.odoms
            # temp_dict['init_collision_time'] = self.init_collision_time
            path = 'rollout_set/'
            path = path + str(len(os.listdir(path))) +'.bag'
            # save(path, temp_dict)
            bag = rosbag.Bag(path, 'w')
            for i in self.odoms:
                bag.write('odometry', i)
            bag.close()
            print ('rollout saved: '+ path)
            self.saved = True
    def land(self, msg):
        print 'land'
        self.completed = True

    def init_collision(self, msg):
        print 'init collision'
        if self.init_collision_time == -1:
            self.init_collision_time = len(self.odoms)

    def takeoff(self, msg):
        print 'takeoff'
        self.completed = False

    def start(self, msg):
        print 'start'
        self.saved = False
        self.terminated = False
        self.sequences = []
        self.velocities = []
        self.odoms = []
    
    def update_vel(self, msg):
        IPython.embed()
        cur_time = time.time()
        delta_t = cur_time - self.last_pos_time
        self.last_pos_time = cur_time
        self.cur_vel = msg.twist.twist.linear
        quat = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x,
         msg.pose.pose.orientation.y, msg.pose.pose.orientation.z]
        cur_angular_z = quad_to_yaw(quat) % (2*math.pi)
        v_angular_z = (cur_angular_z - self.cur_angular_z)/0.2  
        self.cur_angular_z = cur_angular_z
        # if v_angular_z > 1.5:
        #     v_angular_z = 1.5
        # elif v_angular_z < -1.5:
        #     v_angular_z = -1.5
        self.velocities.append([self.cur_vel.x, self.cur_vel.y,v_angular_z])
        self.odoms.append(msg)
        if not self.completed and not self.just_crashed and len(self.velocities)> 8:
            crashed = False
            v1 = np.mean(self.velocities[-3:-2], axis = 0)[0]
            v2 = self.velocities[-1][0]
            # if v1 > 0.45 and v2 < 0.05:
            #     crashed = True
            #     print "front crash {0} vs. {1}".format(v1, v2)
            # temp = np.mean(self.velocities[-6:-2], axis = 0)[2]% math.pi 
            # if temp> self.velocities[-1][2]%math.pi > 0.3 and self.des_control.angular.z > -0.6:
            #     crashed = True
            #     print "side crash {0} vs. {1}".format(temp, self.velocities[-1][2])
            # elif temp > self.velocities[-1][2]%math.pi < -0.3 and self.des_control.angular.z < 0.6:
            #     crashed = True
            #     print "side crash {0} vs. {1}".format(temp, self.velocities[-1][2])
            # if crashed:
            #     self.pub_crash.publish(Empty())

    def evolve_recovery(self):
        if self.recovery_index < len(self.recovery_commands):
            self.cur_control = self.recovery_commands[self.recovery_index]
            self.recovery_index += 1
        else:
            self.recovery_index = 0
            self.just_crashed = False
            print "recovery completed"
            self.cur_control = Twist()
            self.start_recovery = False
            self.pub_start.publish(Empty())

    def update_control(self, msg):
        self.des_control  = msg
        self.last_control_time = time.time()

    def control(self):
        rospy.init_node('bebop_crash_detector', anonymous = True)
        pub_control = rospy.Publisher('/vservo/cmd_vel', Twist, queue_size = 10)
        sub_terminate = rospy.Subscriber('/bebop/collision', Empty, self.terminate)
        sub_terminate = rospy.Subscriber('/bebop/initial_collision', Empty, self.init_collision)
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
                just_started = False
                while not self.terminated:
                    just_started = True
                    r.sleep()
                # 'start recovering'
                # while  self.start_recovery:
                #         self.evolve_recovery()
                #         pub_control.publish(self.cur_control)
                #         r.sleep()
                if just_started:
                    if len(self.velocities) > 10:
                        ys.append(self.velocities[-20:])
                    just_started = False

            if len(ys) != 0:
                f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
                for i in xrange(len(ys)):
                    temp = np.array(ys[i])
                    ax1.plot(temp[:, 0], '-o')
                    ax2.plot(temp[:, 1], '-o')
                    ax3.plot(temp[:, 2], '-o')
                plt.show()
                IPython.embed()
    
def moving_average(data, n = 2):
    ret = np.cumsum(data, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n -1:]/n

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