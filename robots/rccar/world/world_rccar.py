import threading, time

import numpy as np

import rospy, rosbag, rostopic
import std_msgs

try:
    import bair_car.srv 
except:
    pass

import general.ros.ros_utils as ros_utils

from general.world.world import World

from general.state_info.sample import Sample

from config import params

class WorldRCcar(World):

    def __init__(self, agent, bag_file_func, wp=None):
        self._agent = agent
        self.bag = None
        self.bag_lock = threading.Lock()
        self.bag_file_func = bag_file_func
        World.__init__(self, wp=wp)

        self.randomize = self.wp['randomize']

        rccar_topics = params['rccar']['topics']
        ### ROS publishers
        self._ros_reset_pub = rospy.Publisher(rccar_topics['reset'], std_msgs.msg.Empty, queue_size = 100)
        self._cmd_steer_pub = rospy.Publisher(rccar_topics['cmd_steer'], std_msgs.msg.Float32, queue_size=10)
        self._cmd_vel_pub = rospy.Publisher(rccar_topics['cmd_vel'], std_msgs.msg.Float32, queue_size=10)
        ### ROS subscribers
        self._ros_collision = ros_utils.RosCallbackEmpty(rccar_topics['collision'], std_msgs.msg.Empty)
        self._ros_collision_sub = rospy.Subscriber(rccar_topics['collision'],
                                                   std_msgs.msg.Empty,
                                                   callback=self._ros_collision_callback)
        self.num_collisions = 0
        # self._ros_accel = ros_utils.RosCallbackMostRecent(rccar_topics['accel'], gm_msgs.Vector3)
        for topic in rccar_topics.values():
            rostype = rostopic.get_topic_class(topic, blocking=False)[0]
            if rostype:
                rospy.Subscriber(topic, rostype, callback=self._bag_callback, callback_args=(topic,))

    def reset(self, back_up, itr=None, cond=None, rep=None, record=True):
        self._agent.execute_control(None) # stop the car
        assert(itr is not None and cond is not None and rep is not None)

        # TODO add sim backup
        ### back car up straight and slow
        if back_up:
            self._logger.info('Backing the car up')

            sample = Sample(meta_data=params, T=2)
            sample.set_U([np.random.uniform(*self.wp['back_up']['cmd_steer'])], t=0, sub_control='cmd_steer')
            sample.set_U([self.wp['back_up']['cmd_vel']], t=0, sub_control='cmd_vel')
            u = sample.get_U(t=0)

            start = time.time()
            while time.time() - start < self.wp['back_up']['duration']:
                self._agent.execute_control(u)
                time.sleep(0.1)

            self._agent.execute_control(None)

            time.sleep(1.0)

        ### record
        if record:
            self._start_record_bag(itr, cond, rep)
        
        ### TODO add a flag
        self._ros_reset_pub.publish(std_msgs.msg.Empty())
        
        if not self._agent.sim:
            ### reset indicators
            self._ros_collision.get()
            self.num_collisions = 0

    def is_collision(self, sample, t=None):
        #return False
        if self._agent.sim:
            return self._agent.sim_coll
        else:
            return self._ros_collision.get() is not None

    def update_visualization(self, history_sample, planned_sample, t):
        pass

    def get_image(self, sample):
        return None

    def get_info(self):
        self._stop_record_bag()
        ### reset indicators
        self._ros_collision.get()
        return dict()

    ###########
    ### ROS ###
    ###########

    def _start_record_bag(self, itr, cond, rep):
        self.bag = rosbag.Bag(self.bag_file_func(itr, cond, rep), 'w')

    def _bag_callback(self, msg, args):
        topic = args[0]
        with self.bag_lock:
            if self.bag:
                self.bag.write(topic, msg)

    def _stop_record_bag(self):
        if self.bag is None:
            return

        with self.bag_lock:
            bag = self.bag
            self.bag = None
        # bag.reindex()
        bag.close()

    def _ros_collision_callback(self, msg):
        ### immediately stop the car if it's the first collision
        if self._agent.sim_coll:
            if self.num_collisions == 0:
                self._agent.execute_control(None)
            self.num_collisions += 1
