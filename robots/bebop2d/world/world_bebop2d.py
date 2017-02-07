import threading

import rospy, rosbag, rostopic
import std_msgs.msg as std_msgs

import general.ros.ros_utils as ros_utils

from general.world.world import World

from config import params

class WorldBebop2d(World):

    def __init__(self, bag_file_func, wp=None):
        self.bag = None
        self.bag_lock = threading.Lock()
        self.bag_file_func = bag_file_func
        World.__init__(self, wp=wp)

        self.randomize = self.wp['randomize']

        ### ROS subscribers
        self._ros_collision = ros_utils.RosCallbackEmpty(params['bebop']['topics']['collision'], std_msgs.Empty)
        self._ros_start_rollout = ros_utils.RosCallbackEmpty(params['bebop']['topics']['start_rollout'], std_msgs.Empty)
        for topic in params['bebop']['topics'].values():
            rostype = rostopic.get_topic_class(topic, blocking=False)[0]
            if rostype:
                rospy.Subscriber(topic, rostype, callback=self._bag_callback, callback_args=(topic,))

    def reset(self, itr=None, cond=None, rep=None, record=True):
        assert(itr is not None and cond is not None and rep is not None)

        ### wait for user to say start
        self._logger.info('Press start')
        while self._ros_start_rollout.get() is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        self._logger.info('Starting')

        ### record
        if record:
            self._start_record_bag(itr, cond, rep)

        ### reset indicators
        self._ros_start_rollout.get()
        self._ros_collision.get()

    def is_collision(self, sample, t=None):
        return self._ros_collision.get() is not None

    def update_visualization(self, history_sample, planned_sample, t):
        pass # TODO

    def get_image(self, sample):
        return None

    def get_info(self):
        self._stop_record_bag()
        ### reset indicators
        self._ros_start_rollout.get()
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
