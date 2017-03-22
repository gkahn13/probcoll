import threading
from config import params
import rospy, rosbag, rostopic
import std_msgs.msg as std_msgs
import nav_msgs.msg as nav_msgs
import geometry_msgs.msg as geometry_msgs
import IPython
import numpy as np
from robots.bebop2d.ros.state_utils import quad_to_yaw, yaw_to_quad
import general.ros.ros_utils as ros_utils
from general.world.world import World
import math
import Queue

class CrashDetector():
    def __init__(self, rate=20):
        rospy.init_node('CrashDetector', anonymous=True)
        self._ros_collision = rospy.Publisher('bebop/collision', std_msgs.Empty)
        self._ros_start_rollout = rospy.Publisher('bebop/start_rollout', std_msgs.Empty)
        self._ros_cmd = ros_utils.RatePublisher(10., 'vservo/cmd_vel', geometry_msgs.Twist)
        self._ros_odom = ros_utils.RosCallbackMostRecent('bebop/odom', nav_msgs.Odometry)
        self._ros_started = ros_utils.RosCallbackMostRecent('bebop/start_rollout', std_msgs.Empty)
        self.rate = rospy.Rate(rate)

        # drone state
        self.started = False
        self.last_pos = None
        self.last_yaw = None

        #
        self.pos_queue = Queue.Queue(maxsize=5)
        self.yaw_queue = Queue.Queue(maxsize=5)
    def publish_crash(self):
        self._ros_collision.publish(std_msgs.Empty())
    def start_rollout(self):
        self._ros_start_rollout.publish(std_msgs.Empty())
    def run(self):
        # i = 0
        # j = 0
        while not rospy.is_shutdown():
            last_dom_msg = self._ros_odom.get()
            # print i
            # i +=1
            # j +=1
            if self._ros_started.get() is not None:
                self.started = True
                self.pos_queue.queue.clear()
                self.yaw_queue.queue.clear()
            if self.started and last_dom_msg is not None:
                self.last_pos = [last_dom_msg.pose.pose.position.x,\
                                last_dom_msg.pose.pose.position.y, last_dom_msg.pose.pose.position.z]
                # IPython.embed()
                quaternion = [last_dom_msg.pose.pose.orientation.w, last_dom_msg.pose.pose.orientation.x,\
                              last_dom_msg.pose.pose.orientation.y, last_dom_msg.pose.pose.orientation.z]
                # self.last_ori = quad_to_yaw(quaternion)
                if self.pos_queue.full():
                    self.pos_queue.get()
                    self.yaw_queue.get()
                self.pos_queue.put(self.last_pos)
                self.yaw_queue.put(self.last_yaw)
                # print j
                if self.pos_queue.full() and self.isCrash():
                    self.publish_crash()
                    print "Crash detected!"
                    self.started = False
                    self.pos_queue.queue.clear()
                    self.yaw_queue.queue.clear()
                    self.adjust_position()
                    print "Adjustment complete! Continue exploration!"
                    # self.start_rollout()
            self.rate.sleep()

    def isCrash(self):
        positions = []
        # yaws = []
        for i in range(self.pos_queue.qsize()):
            positions.append(self.pos_queue.queue[i])
            # yaws.append(self.yaw_queue.queue[i])
        mean_pos = np.mean(np.array(positions), axis=0)
        if np.linalg.norm(mean_pos - self.last_pos) < 0.1:
            self.crashed = True
            return True
        return False
    def adjust_position(self):
        back_up_command = geometry_msgs.Twist(geometry_msgs.Vector3(-.5,0,0),geometry_msgs.Vector3(0,0,0))
        rotate_command = geometry_msgs.Twist(geometry_msgs.Vector3(0,0,0),geometry_msgs.Vector3(0,0,.6))
        for _ in xrange(20):
            # IPython.embed()
            self._ros_cmd.publish(back_up_command)
            self.rate.sleep()
        for _ in xrange(30):
            self._ros_cmd.publish(rotate_command)
            self.rate.sleep()
        self._ros_cmd.publish(geometry_msgs.Twist())




if __name__ == '__main__':
    crashDetector = CrashDetector()
    crashDetector.run()
    IPython.embed()