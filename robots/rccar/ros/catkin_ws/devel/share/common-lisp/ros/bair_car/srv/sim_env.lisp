; Auto-generated. Do not edit!


(cl:in-package bair_car-srv)


;//! \htmlinclude sim_env-request.msg.html

(cl:defclass <sim_env-request> (roslisp-msg-protocol:ros-message)
  ((steer
    :reader steer
    :initarg :steer
    :type cl:float
    :initform 0.0)
   (motor
    :reader motor
    :initarg :motor
    :type cl:float
    :initform 0.0)
   (vel
    :reader vel
    :initarg :vel
    :type cl:float
    :initform 0.0)
   (reset
    :reader reset
    :initarg :reset
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass sim_env-request (<sim_env-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <sim_env-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'sim_env-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name bair_car-srv:<sim_env-request> is deprecated: use bair_car-srv:sim_env-request instead.")))

(cl:ensure-generic-function 'steer-val :lambda-list '(m))
(cl:defmethod steer-val ((m <sim_env-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader bair_car-srv:steer-val is deprecated.  Use bair_car-srv:steer instead.")
  (steer m))

(cl:ensure-generic-function 'motor-val :lambda-list '(m))
(cl:defmethod motor-val ((m <sim_env-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader bair_car-srv:motor-val is deprecated.  Use bair_car-srv:motor instead.")
  (motor m))

(cl:ensure-generic-function 'vel-val :lambda-list '(m))
(cl:defmethod vel-val ((m <sim_env-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader bair_car-srv:vel-val is deprecated.  Use bair_car-srv:vel instead.")
  (vel m))

(cl:ensure-generic-function 'reset-val :lambda-list '(m))
(cl:defmethod reset-val ((m <sim_env-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader bair_car-srv:reset-val is deprecated.  Use bair_car-srv:reset instead.")
  (reset m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <sim_env-request>) ostream)
  "Serializes a message object of type '<sim_env-request>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'steer))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'motor))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'vel))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'reset) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <sim_env-request>) istream)
  "Deserializes a message object of type '<sim_env-request>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'steer) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'motor) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'vel) (roslisp-utils:decode-single-float-bits bits)))
    (cl:setf (cl:slot-value msg 'reset) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<sim_env-request>)))
  "Returns string type for a service object of type '<sim_env-request>"
  "bair_car/sim_envRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'sim_env-request)))
  "Returns string type for a service object of type 'sim_env-request"
  "bair_car/sim_envRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<sim_env-request>)))
  "Returns md5sum for a message object of type '<sim_env-request>"
  "b309b81746c8edad7c966dda272a1ed4")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'sim_env-request)))
  "Returns md5sum for a message object of type 'sim_env-request"
  "b309b81746c8edad7c966dda272a1ed4")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<sim_env-request>)))
  "Returns full string definition for message of type '<sim_env-request>"
  (cl:format cl:nil "~%float32 steer~%float32 motor~%float32 vel~%bool reset~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'sim_env-request)))
  "Returns full string definition for message of type 'sim_env-request"
  (cl:format cl:nil "~%float32 steer~%float32 motor~%float32 vel~%bool reset~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <sim_env-request>))
  (cl:+ 0
     4
     4
     4
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <sim_env-request>))
  "Converts a ROS message object to a list"
  (cl:list 'sim_env-request
    (cl:cons ':steer (steer msg))
    (cl:cons ':motor (motor msg))
    (cl:cons ':vel (vel msg))
    (cl:cons ':reset (reset msg))
))
;//! \htmlinclude sim_env-response.msg.html

(cl:defclass <sim_env-response> (roslisp-msg-protocol:ros-message)
  ((coll
    :reader coll
    :initarg :coll
    :type cl:boolean
    :initform cl:nil)
   (image
    :reader image
    :initarg :image
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image))
   (depth
    :reader depth
    :initarg :depth
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image))
   (pose
    :reader pose
    :initarg :pose
    :type geometry_msgs-msg:Pose
    :initform (cl:make-instance 'geometry_msgs-msg:Pose)))
)

(cl:defclass sim_env-response (<sim_env-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <sim_env-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'sim_env-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name bair_car-srv:<sim_env-response> is deprecated: use bair_car-srv:sim_env-response instead.")))

(cl:ensure-generic-function 'coll-val :lambda-list '(m))
(cl:defmethod coll-val ((m <sim_env-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader bair_car-srv:coll-val is deprecated.  Use bair_car-srv:coll instead.")
  (coll m))

(cl:ensure-generic-function 'image-val :lambda-list '(m))
(cl:defmethod image-val ((m <sim_env-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader bair_car-srv:image-val is deprecated.  Use bair_car-srv:image instead.")
  (image m))

(cl:ensure-generic-function 'depth-val :lambda-list '(m))
(cl:defmethod depth-val ((m <sim_env-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader bair_car-srv:depth-val is deprecated.  Use bair_car-srv:depth instead.")
  (depth m))

(cl:ensure-generic-function 'pose-val :lambda-list '(m))
(cl:defmethod pose-val ((m <sim_env-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader bair_car-srv:pose-val is deprecated.  Use bair_car-srv:pose instead.")
  (pose m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <sim_env-response>) ostream)
  "Serializes a message object of type '<sim_env-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'coll) 1 0)) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'image) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'depth) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'pose) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <sim_env-response>) istream)
  "Deserializes a message object of type '<sim_env-response>"
    (cl:setf (cl:slot-value msg 'coll) (cl:not (cl:zerop (cl:read-byte istream))))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'image) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'depth) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'pose) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<sim_env-response>)))
  "Returns string type for a service object of type '<sim_env-response>"
  "bair_car/sim_envResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'sim_env-response)))
  "Returns string type for a service object of type 'sim_env-response"
  "bair_car/sim_envResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<sim_env-response>)))
  "Returns md5sum for a message object of type '<sim_env-response>"
  "b309b81746c8edad7c966dda272a1ed4")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'sim_env-response)))
  "Returns md5sum for a message object of type 'sim_env-response"
  "b309b81746c8edad7c966dda272a1ed4")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<sim_env-response>)))
  "Returns full string definition for message of type '<sim_env-response>"
  (cl:format cl:nil "~%bool coll~%sensor_msgs/Image image~%sensor_msgs/Image depth~%geometry_msgs/Pose pose~%~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of cameara~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'sim_env-response)))
  "Returns full string definition for message of type 'sim_env-response"
  (cl:format cl:nil "~%bool coll~%sensor_msgs/Image image~%sensor_msgs/Image depth~%geometry_msgs/Pose pose~%~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of cameara~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <sim_env-response>))
  (cl:+ 0
     1
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'image))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'depth))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'pose))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <sim_env-response>))
  "Converts a ROS message object to a list"
  (cl:list 'sim_env-response
    (cl:cons ':coll (coll msg))
    (cl:cons ':image (image msg))
    (cl:cons ':depth (depth msg))
    (cl:cons ':pose (pose msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'sim_env)))
  'sim_env-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'sim_env)))
  'sim_env-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'sim_env)))
  "Returns string type for a service object of type '<sim_env>"
  "bair_car/sim_env")