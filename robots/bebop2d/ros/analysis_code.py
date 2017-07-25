import rospy
import numpy as np
from std_msgs.msg import Empty
from geometry_msgs.msg import Vector3, Point, Twist, Pose
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from transformations import euler_from_quaternion
import time
import IPython
import sys
import math
from state_utils import quad_to_yaw
import matplotlib.pyplot as plt
import pickle
import os
import rosbag


topics = ['/bebop/odom', 'vservo/cmd_vel']

class CrashDetector():
    def __init__(self):
        self.prev_states = []
        self.cur_state = None
    def predict(self, cur_state):
        self.prev_states.append(cur_state)
        if len(self.prev_states) < 6:
            return False
        last_state = self.prev_states[-2]
        second_last_state = self.prev_states[-3]
        t_linear_x, t_linear_y, t_des_linear_x, t_des_linear_y, t_pos_z, t_angular_x, t_angular_y, t_angular_z = cur_state
        t_b1_linear_x, t_b1_linear_y, t_b1_des_linear_x, t_b1_des_linear_y, t_b1_pos_z, t_b1_angular_x, t_b1_angular_y, t_b1_angular_z = last_state
        t_b2_linear_x, t_b2_linear_y, t_b2_des_linear_x, t_b2_des_linear_y, t_b2_pos_z, t_b2_angular_x, t_b2_angular_y, t_b2_angular_z = second_last_state
        t_b3_linear_x, t_b3_linear_y, t_b3_des_linear_x, t_b3_des_linear_y, t_b3_pos_z, t_b3_angular_x, t_b3_angular_y, t_b3_angular_z = self.prev_states[-4]
        t_b4_linear_x, t_b4_linear_y, t_b4_des_linear_x, t_b4_des_linear_y, t_b4_pos_z, t_b4_angular_x, t_b4_angular_y, t_b4_angular_z = self.prev_states[-5]
        t_b5_linear_x, t_b5_linear_y, t_b5_des_linear_x, t_b5_des_linear_y, t_b5_pos_z, t_b5_angular_x, t_b5_angular_y, t_b5_angular_z = self.prev_states[-6]
        if t_linear_x - t_b1_linear_x < -0.2:
            return True
        # elif cur_linear_x - last_linear_x < -0.1:
        #      if abs((cur_linear_y - last_linear_y) - (last_linear_y - s_last_linear_y)) > 0.1:
        #         return True

        delta_c_x = t_b2_des_linear_x - t_linear_x 
        # delta_c_y =  t_b3_des_linear_y - t_linear_y if t_b3_des_linear_y > 0 else t_linear_y - t_b3_des_linear_y
        delta_c_y =  abs(t_b2_des_linear_y - t_linear_y)
        delta_l_x = t_b3_des_linear_x - t_b1_linear_x 
        delta_l_y = abs(t_b3_des_linear_y - t_b1_linear_y)
        # delta_l_y = t_b4_des_linear_y - t_b1_linear_y if t_b4_des_linear_y > 0 else t_b1_linear_y - t_b4_des_linear_y
        delta_sl_x = t_b4_des_linear_x - t_b2_linear_x 
        # delta_sl_y = t_b5_des_linear_y - t_b2_linear_y if t_b5_des_linear_y > 0 else t_b2_linear_y - t_b5_des_linear_y
        delta_sl_y = abs(t_b4_des_linear_y - t_b2_linear_y)
        if t_b2_des_linear_y == t_b4_des_linear_y and t_b4_des_linear_y == t_b3_des_linear_y and delta_c_y > delta_l_y and delta_l_y > delta_sl_y and delta_c_y > 0.35:
            return True


        # delta_sl_x = s_last_des_linear_x - s_last_linear_x 
        # delta_sl_y = s_last_des_linear_y - last_linear_y if last_des_linear_y > 0 else s_last_linear_y - s_last_des_linear_y
        # if delta_c_x > 0.2 and delta_l_x > 0.1 and delta_sl_x < 0:
        #     return True
        # if delta_c_x > delta_l_x and delta_l_x > delta_sl_x and delta_c_x - delta_sl_x > 0.2 and delta_sl_x < 0:
        #     return True
        # ind_y = abs(change(cur_angular_y, last_angular_y) - change(last_angular_y, s_last_angular_y)) > 0.15
        # ind_x = abs(change(cur_angular_x, last_angular_x) - change(last_angular_x, s_last_angular_x)) > 0.15
        # ind_z = abs(change(cur_angular_z, last_angular_z) - change(last_angular_z, s_last_angular_z)) > 0.15
        # if ind_x:
        #     return True
        return False

def change(x1, x2):
    x1 = x1%math.pi
    x2 = x2%math.pi
    if abs(x1 - x2) > math.pi/2:
       if x1 > math.pi/2:
            change = x2 + math.pi - x1
       else: 
            change = x1  + math.pi - x2
    else:
        change = x1 - x2
    return change
def analyze(plot_prediction = False):
    data = load()
    total_num_of_traj = len(data)
    hist = 20*[0.0]
    total_predicted = 0
    iter = 0
    failed_rollouts = []
    for (datum, collision_time), rollout_name in data:
        iter += 1
        linear_x, linear_y,cur_speed_x, cur_speed_y, pos_z, angular_x, angular_y, angular_z = datum
        zipped_datum = zip(datum[0], datum[1], datum[2], datum[3], datum[4], datum[5], datum[6], datum[7])
        detector = CrashDetector()
        i = 10 - collision_time
        detected_collision_time = -1
        predicted = False
        for cur_state in zipped_datum:
            just_prpedicted = False
            if detector.predict(cur_state):
                if i < 10:
                    print 'detected before happening'
                hist[i] += 1
                detected_collision_time = i
                total_predicted += 1
                predicted = True
                break
                if not just_prpedicted:
                    # print 'detected crash'
                    just_prpedicted = True
            else:
                just_prpedicted = False
            i += 1
        if not predicted:
            # print 'no crash'
            failed_rollouts.append(rollout_name)
        if plot_prediction:
            plt.figure()
            f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True)
            for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
                ax.axvline(collision_time-0.5, color='r', linestyle='--')
                if detected_collision_time != -1:
                    ax.axvline(detected_collision_time, color = 'b', linestyle='dotted')
            ax1.plot(linear_x, '-o')
            ax2.plot(linear_y, '-o')
            ax1.plot(linear_x[:3] + cur_speed_x[:-3], '--', color='r')
            ax2.plot(linear_y[:3] + cur_speed_y[:-3], '--', color='r')
            # ax2.plot(cur_speed_y, '--', color='r')
            ax3.plot(pos_z, '-o')
            ax4.plot(angular_x, '-o')
            ax5.plot(angular_y, '-o')
            ax6.plot(angular_z, '-o')
            plt.savefig('prediction_plots/'+ rollout_name +'.jpg')
            plt.close(f)
    print failed_rollouts
    print "predicted {0} out of {1}".format(total_predicted, total_num_of_traj)
    plt.figure()
    plt.plot(np.divide(np.cumsum(hist), len(data)))
    plt.axvline(9.5, color='r', linestyle='--')
    plt.savefig('prediction_result.jpg')

def process_bag(bag): 
    collisions = list(bag.read_messages(topics=['/bebop/collision']))
    collision_time = collisions[0].timestamp
    # collision_time = list(bag.read_messages(topics=['/bebop/odom']))[-1].timestamp
    for c in collisions:
        if c.timestamp < collision_time:
            collision_time = c.timestamp
    cur_speeds = []
    before_collision_odoms = []
    after_collision_odoms = []
    last_speed = None
    for topic, msg, t in bag.read_messages(topics=['/bebop/odom', '/vservo/cmd_vel']):
        if topic == '/vservo/cmd_vel':
            if len(before_collision_odoms) + len(after_collision_odoms) > len(cur_speeds):
                cur_speeds += [msg]*(len(before_collision_odoms) + len(after_collision_odoms)-len(cur_speeds))
            else:
                last_speed = msg
        elif topic == '/bebop/odom':
            if t < collision_time:
                before_collision_odoms.append(msg)
            else:
                after_collision_odoms.append(msg)
            if last_speed:
                cur_speeds.append(last_speed)
    # print "{0}, {1}, {2}".format(len(before_collision_odoms), len(after_collision_odoms), len(cur_speeds))
    return before_collision_odoms, after_collision_odoms,cur_speeds

def load():
    data = []
    path = 'rollout_set/'
    for dir in os.listdir(path):
        bag = rosbag.Bag(path + dir)
        data.append([collect_info_from_bag(process_bag(bag), False, dir), dir])
    return data

def collect_info_from_bag(bag, plot_traj=False, rollout_name=None):

    # truncate if necessary
    before_collision_odoms, after_collision_odoms, cur_speeds = bag
    before_collision_speeds = cur_speeds[:len(before_collision_odoms)]
    after_collision_speeds = cur_speeds[len(before_collision_odoms):]
    if len(before_collision_odoms) > 10:
        before_collision_odoms = before_collision_odoms[-10:]
        before_collision_speeds = before_collision_speeds[-10:]
    if len(after_collision_odoms) > 10:
        after_collision_odoms = after_collision_odoms[10:]
        after_collision_speeds = after_collision_speeds[10:]

    # read odoms and append to list
    linear_x = []
    linear_y = []
    cur_speed_x = []
    cur_speed_y = []
    pos_z = []
    angular_x = []
    angular_y = []
    angular_z = []

    for i in xrange(len(before_collision_odoms)):
        msg = before_collision_odoms[i]
        speed_msg = before_collision_speeds[i]
        linear_x.append(msg.twist.twist.linear.x)
        linear_y.append(msg.twist.twist.linear.y)
        cur_speed_x.append(speed_msg.linear.x)
        cur_speed_y.append(speed_msg.linear.y)
        pos_z.append(msg.pose.pose.position.z)
        quat = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y, msg.pose.pose.orientation.z]
        x, y, z = euler_from_quaternion(quat)
        angular_x.append(x % math.pi)
        angular_z.append(z % math.pi)
        angular_y.append(y % math.pi)
    for i in xrange(len(after_collision_odoms)):
        msg = after_collision_odoms[i]
        speed_msg = after_collision_speeds[i]
        linear_x.append(msg.twist.twist.linear.x)
        linear_y.append(msg.twist.twist.linear.y)
        cur_speed_x.append(speed_msg.linear.x)
        cur_speed_y.append(speed_msg.linear.y)
        pos_z.append(msg.pose.pose.position.z)
        quat = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y, msg.pose.pose.orientation.z]
        x, y, z = euler_from_quaternion(quat)
        angular_x.append(x % math.pi)
        angular_z.append(z % math.pi)
        angular_y.append(y % math.pi)

    if plot_traj:
        plt.figure()
        f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True)
        for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
            ax.axvline(len(before_collision_odoms)-0.5, color='r', linestyle='--')
        ax1.plot(linear_x, '-o')
        ax2.plot(linear_y, '-o')
        ax1.plot(cur_speed_x, '--', color='r')
        ax2.plot(cur_speed_y, '--', color='r')
        ax3.plot(pos_z, '-o')
        ax4.plot(angular_x, '-o')
        ax5.plot(angular_y, '-o')
        ax6.plot(angular_z, '-o')
        plt.savefig('plots/'+ rollout_name +'.jpg')
    return (linear_x, linear_y,cur_speed_x, cur_speed_y, pos_z, angular_x, angular_y, angular_z), len(before_collision_odoms)

# def process():
#     bags = []
#     path = 'rollout_set/'
#     for dir in os.listdir(path):
#         bag = rosbag.Bag(path + dir)
#         temp = []
#         for topic, msg, t in bag.read_messages(topics=topics):
#             temp.append(msg)
#         bags.append(temp[-10:])
#     data = []
#     for i, bag in enumerate(bags):
#         linear_x = []
#         linear_y = []
#         linear_z = []
#         angular_x = []
#         angular_y = []
#         angular_z = []
#         for msg in bag:
#             linear_x.append(msg.twist.twist.linear.x)
#             linear_y.append(msg.twist.twist.linear.y)
#             quat = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x,
#             msg.pose.pose.orientation.y, msg.pose.pose.orientation.z]
#             x, y, z = euler_from_quaternion(quat)
#             angular_x.append(x % math.pi)
#             angular_z.append(z % math.pi)
#             angular_y.append(y % math.pi)
#         # plt.figure()
#         # f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True)
#         # ax1.plot(linear_x, '-o')
#         # ax2.plot(linear_y, '-o')
#         # ax3.plot(angular_x, '-o')
#         # ax4.plot(angular_y, '-o')
#         # ax5.plot(angular_z, '-o')
#         # plt.savefig('plots/'+str(i)+'.jpg')
#         data.append([linear_x, linear_y, angular_x, angular_y, angular_z])
#     return data
if __name__ == '__main__':
    analyze(False)