import abc
import threading
import rospy
from config import params

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
        new_topic = "/{0}/{1}".format(params['exp_name'], topic)  
        RosCallbackAbstract.__init__(self, new_topic, data_class)
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
        new_topic = "/{0}/{1}".format(params['exp_name'], topic)  
        RosCallbackAbstract.__init__(self, new_topic, data_class)
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

class Publisher:
    
    def __init__(self, topic, msg_type, **kwargs):
        new_topic = "/{0}/{1}".format(params['exp_name'], topic)  
        self.pub = rospy.Publisher(new_topic, msg_type, **kwargs)

    def publish(self, msg, **kwargs):
        return self.pub.publish(msg, **kwargs)

###################
### Subscribers ###
###################

class Subscriber:
    
    def __init__(self, topic, msg_type, **kwargs):
        new_topic = "/{0}/{1}".format(params['exp_name'], topic)  
        self.sub = rospy.Subscriber(new_topic, msg_type, **kwargs)

###################
### Services ###
###################

def ServiceProxy(service, env, **kwargs):
    new_service = "/{0}/{1}".format(params['exp_name'], service)  
    return rospy.ServiceProxy(new_service, env, **kwargs)

def wait_for_service(service, **kwargs):
    new_service = "/{0}/{1}".format(params['exp_name'], service)  
    return rospy.wait_for_service(new_service, **kwargs)

