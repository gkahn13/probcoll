import abc
import rospy
import sensor_msgs.msg
import cv_bridge
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
### ROS Image   ###
###################

class ROSListener:
    def __init__(self, topic, msg_type):
        self.sub = rospy.Subscriber(topic, msg_type, callback=self._callback)
        self.msg = None
        
    def _callback(self, msg):
        self.msg = msg

    def get_msg(self):
        return self.msg

class ImageROSListener(object):
    def __init__(self, topic, msg_type=sensor_msgs.msg.Image):
        self.ros_listener = ROSListener(topic, msg_type)
        self.bridge = cv_bridge.CvBridge()

    def get_image(self):
        msg = self.ros_listener.get_msg()

        if msg is None:
            return None

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except cv_bridge.CvBridgeError as e:
            return None

        return cv_image

class ImageROSPublisher(object):
    def __init__(self, topic, msg_type=sensor_msgs.msg.Image):
        self.bridge = cv_bridge.CvBridge()
	self.publisher = rospy.Publisher(topic, msg_type, queue_size=1)	
    def publish_image(self, cv_image, image_format="rgb8"):
    	try:
	    ros_image = self.bridge.cv2_to_imgmsg(cv_image, image_format)
	    self.publisher.publish(ros_image)
	except CvBridgeError as e:
	    pass

class ImageROSHandler(ImageROSListener):
    def __init__(self, sub_topic, pub_topic, msg_type=sensor_msgs.msg.Image):
	super(ImageROSHandler, self).__init__(sub_topic, msg_type)
	self.publisher = rospy.Publisher(pub_topic, msg_type, queue_size=1)	

    def publish_image(self, cv_image):
    	try:
	    ros_image = self.bridge.cv2_to_imgmsg(cv_image, "rgb8")
	    self.publisher.publish(ros_image)
	except CvBridgeError as e:
	    pass
