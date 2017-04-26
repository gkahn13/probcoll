import rospy

import sensor_msgs.msg
import cv_bridge

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
