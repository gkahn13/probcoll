import abc

import threading

import rospy

class RosCallbackAbstract(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, topic, data_class):
        self.sub = rospy.Subscriber(topic, data_class, callback=self._callback)

    @abc.abstractmethod
    def _callback(self, msg):
        raise NotImplementedError('Implement in subclass')

    @abc.abstractmethod
    def get(self):
        raise NotImplementedError('Implement in subclass')

class RosCallbackMostRecent(RosCallbackAbstract):
    def __init__(self, topic, data_class, clear_on_get=True):
        RosCallbackAbstract.__init__(self, topic, data_class)
        self.clear_on_get = clear_on_get
        self.msg = None

    def _callback(self, msg):
        self.msg = msg

    def get(self):
        msg = self.msg
        if self.clear_on_get:
            self.msg = None
        return msg

class RosCallbackAll(RosCallbackAbstract):
    def __init__(self, topic, data_class, max_num_msgs=100, clear_msgs=True):
        RosCallbackAbstract.__init__(self, topic, data_class)
        self.msgs = []
        self.max_num_msgs = max_num_msgs
        self.clear_msgs = clear_msgs

    def _callback(self, msg):
        self.msgs.append(msg)
        if len(self.msgs) > self.max_num_msgs:
            self.msgs = self.msgs[-self.max_num_msgs:]

    def get(self):
        msgs = self.msgs
        if self.clear_msgs:
            self.msgs = []
        return msgs

##############################
### Specific message types ###
##############################

class RosCallbackEmpty(RosCallbackMostRecent):

    def _callback(self, msg):
        self.msg = rospy.Time.now()

##################
### Publishers ###
##################

class RatePublisher(rospy.Publisher):
    def __init__(self, rate, name, data_class,
                 subscriber_listener=None, tcp_nodelay=False, latch=False, headers=None, queue_size=None):
        self.rate = rospy.Rate(rate)
        self.msg = None

        rospy.Publisher.__init__(self, name, data_class,
                                 subscriber_listener=subscriber_listener,
                                 tcp_nodelay=tcp_nodelay,
                                 latch=latch,
                                 headers=headers,
                                 queue_size=queue_size)

        threading.Thread(target=self._run).start()

    def publish(self, msg):
        self.msg = msg

    def _run(self):
        while not rospy.is_shutdown():
            if self.msg is not None:
                super(RatePublisher, self).publish(self.msg)
            self.rate.sleep()
