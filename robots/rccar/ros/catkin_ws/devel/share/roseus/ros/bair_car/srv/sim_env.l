;; Auto-generated. Do not edit!


(when (boundp 'bair_car::sim_env)
  (if (not (find-package "BAIR_CAR"))
    (make-package "BAIR_CAR"))
  (shadow 'sim_env (find-package "BAIR_CAR")))
(unless (find-package "BAIR_CAR::SIM_ENV")
  (make-package "BAIR_CAR::SIM_ENV"))
(unless (find-package "BAIR_CAR::SIM_ENVREQUEST")
  (make-package "BAIR_CAR::SIM_ENVREQUEST"))
(unless (find-package "BAIR_CAR::SIM_ENVRESPONSE")
  (make-package "BAIR_CAR::SIM_ENVRESPONSE"))

(in-package "ROS")



(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "SENSOR_MSGS"))
  (ros::roseus-add-msgs "sensor_msgs"))


(defclass bair_car::sim_envRequest
  :super ros::object
  :slots (_steer _motor _vel _reset ))

(defmethod bair_car::sim_envRequest
  (:init
   (&key
    ((:steer __steer) 0.0)
    ((:motor __motor) 0.0)
    ((:vel __vel) 0.0)
    ((:reset __reset) nil)
    )
   (send-super :init)
   (setq _steer (float __steer))
   (setq _motor (float __motor))
   (setq _vel (float __vel))
   (setq _reset __reset)
   self)
  (:steer
   (&optional __steer)
   (if __steer (setq _steer __steer)) _steer)
  (:motor
   (&optional __motor)
   (if __motor (setq _motor __motor)) _motor)
  (:vel
   (&optional __vel)
   (if __vel (setq _vel __vel)) _vel)
  (:reset
   (&optional __reset)
   (if __reset (setq _reset __reset)) _reset)
  (:serialization-length
   ()
   (+
    ;; float32 _steer
    4
    ;; float32 _motor
    4
    ;; float32 _vel
    4
    ;; bool _reset
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _steer
       (sys::poke _steer (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _motor
       (sys::poke _motor (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _vel
       (sys::poke _vel (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; bool _reset
       (if _reset (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _steer
     (setq _steer (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _motor
     (setq _motor (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _vel
     (setq _vel (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; bool _reset
     (setq _reset (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass bair_car::sim_envResponse
  :super ros::object
  :slots (_coll _image _depth _pose ))

(defmethod bair_car::sim_envResponse
  (:init
   (&key
    ((:coll __coll) nil)
    ((:image __image) (instance sensor_msgs::Image :init))
    ((:depth __depth) (instance sensor_msgs::Image :init))
    ((:pose __pose) (instance geometry_msgs::Pose :init))
    )
   (send-super :init)
   (setq _coll __coll)
   (setq _image __image)
   (setq _depth __depth)
   (setq _pose __pose)
   self)
  (:coll
   (&optional __coll)
   (if __coll (setq _coll __coll)) _coll)
  (:image
   (&rest __image)
   (if (keywordp (car __image))
       (send* _image __image)
     (progn
       (if __image (setq _image (car __image)))
       _image)))
  (:depth
   (&rest __depth)
   (if (keywordp (car __depth))
       (send* _depth __depth)
     (progn
       (if __depth (setq _depth (car __depth)))
       _depth)))
  (:pose
   (&rest __pose)
   (if (keywordp (car __pose))
       (send* _pose __pose)
     (progn
       (if __pose (setq _pose (car __pose)))
       _pose)))
  (:serialization-length
   ()
   (+
    ;; bool _coll
    1
    ;; sensor_msgs/Image _image
    (send _image :serialization-length)
    ;; sensor_msgs/Image _depth
    (send _depth :serialization-length)
    ;; geometry_msgs/Pose _pose
    (send _pose :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _coll
       (if _coll (write-byte -1 s) (write-byte 0 s))
     ;; sensor_msgs/Image _image
       (send _image :serialize s)
     ;; sensor_msgs/Image _depth
       (send _depth :serialize s)
     ;; geometry_msgs/Pose _pose
       (send _pose :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _coll
     (setq _coll (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; sensor_msgs/Image _image
     (send _image :deserialize buf ptr-) (incf ptr- (send _image :serialization-length))
   ;; sensor_msgs/Image _depth
     (send _depth :deserialize buf ptr-) (incf ptr- (send _depth :serialization-length))
   ;; geometry_msgs/Pose _pose
     (send _pose :deserialize buf ptr-) (incf ptr- (send _pose :serialization-length))
   ;;
   self)
  )

(defclass bair_car::sim_env
  :super ros::object
  :slots ())

(setf (get bair_car::sim_env :md5sum-) "b309b81746c8edad7c966dda272a1ed4")
(setf (get bair_car::sim_env :datatype-) "bair_car/sim_env")
(setf (get bair_car::sim_env :request) bair_car::sim_envRequest)
(setf (get bair_car::sim_env :response) bair_car::sim_envResponse)

(defmethod bair_car::sim_envRequest
  (:response () (instance bair_car::sim_envResponse :init)))

(setf (get bair_car::sim_envRequest :md5sum-) "b309b81746c8edad7c966dda272a1ed4")
(setf (get bair_car::sim_envRequest :datatype-) "bair_car/sim_envRequest")
(setf (get bair_car::sim_envRequest :definition-)
      "
float32 steer
float32 motor
float32 vel
bool reset
---

bool coll
sensor_msgs/Image image
sensor_msgs/Image depth
geometry_msgs/Pose pose


================================================================================
MSG: sensor_msgs/Image
# This message contains an uncompressed image
# (0, 0) is at top-left corner of image
#

Header header        # Header timestamp should be acquisition time of image
                     # Header frame_id should be optical frame of camera
                     # origin of frame should be optical center of cameara
                     # +x should point to the right in the image
                     # +y should point down in the image
                     # +z should point into to plane of the image
                     # If the frame_id here and the frame_id of the CameraInfo
                     # message associated with the image conflict
                     # the behavior is undefined

uint32 height         # image height, that is, number of rows
uint32 width          # image width, that is, number of columns

# The legal values for encoding are in file src/image_encodings.cpp
# If you want to standardize a new string format, join
# ros-users@lists.sourceforge.net and send an email proposing a new encoding.

string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes
uint8[] data          # actual matrix data, size is (step * rows)

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
# 0: no frame
# 1: global frame
string frame_id

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w
")

(setf (get bair_car::sim_envResponse :md5sum-) "b309b81746c8edad7c966dda272a1ed4")
(setf (get bair_car::sim_envResponse :datatype-) "bair_car/sim_envResponse")
(setf (get bair_car::sim_envResponse :definition-)
      "
float32 steer
float32 motor
float32 vel
bool reset
---

bool coll
sensor_msgs/Image image
sensor_msgs/Image depth
geometry_msgs/Pose pose


================================================================================
MSG: sensor_msgs/Image
# This message contains an uncompressed image
# (0, 0) is at top-left corner of image
#

Header header        # Header timestamp should be acquisition time of image
                     # Header frame_id should be optical frame of camera
                     # origin of frame should be optical center of cameara
                     # +x should point to the right in the image
                     # +y should point down in the image
                     # +z should point into to plane of the image
                     # If the frame_id here and the frame_id of the CameraInfo
                     # message associated with the image conflict
                     # the behavior is undefined

uint32 height         # image height, that is, number of rows
uint32 width          # image width, that is, number of columns

# The legal values for encoding are in file src/image_encodings.cpp
# If you want to standardize a new string format, join
# ros-users@lists.sourceforge.net and send an email proposing a new encoding.

string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes
uint8[] data          # actual matrix data, size is (step * rows)

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
# 0: no frame
# 1: global frame
string frame_id

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w
")



(provide :bair_car/sim_env "b309b81746c8edad7c966dda272a1ed4")


