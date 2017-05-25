// Auto-generated. Do not edit!

// (in-package bair_car.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

let geometry_msgs = _finder('geometry_msgs');
let sensor_msgs = _finder('sensor_msgs');

//-----------------------------------------------------------

class sim_envRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.steer = null;
      this.motor = null;
      this.vel = null;
      this.reset = null;
    }
    else {
      if (initObj.hasOwnProperty('steer')) {
        this.steer = initObj.steer
      }
      else {
        this.steer = 0.0;
      }
      if (initObj.hasOwnProperty('motor')) {
        this.motor = initObj.motor
      }
      else {
        this.motor = 0.0;
      }
      if (initObj.hasOwnProperty('vel')) {
        this.vel = initObj.vel
      }
      else {
        this.vel = 0.0;
      }
      if (initObj.hasOwnProperty('reset')) {
        this.reset = initObj.reset
      }
      else {
        this.reset = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type sim_envRequest
    // Serialize message field [steer]
    bufferOffset = _serializer.float32(obj.steer, buffer, bufferOffset);
    // Serialize message field [motor]
    bufferOffset = _serializer.float32(obj.motor, buffer, bufferOffset);
    // Serialize message field [vel]
    bufferOffset = _serializer.float32(obj.vel, buffer, bufferOffset);
    // Serialize message field [reset]
    bufferOffset = _serializer.bool(obj.reset, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type sim_envRequest
    let len;
    let data = new sim_envRequest(null);
    // Deserialize message field [steer]
    data.steer = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [motor]
    data.motor = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [vel]
    data.vel = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [reset]
    data.reset = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 13;
  }

  static datatype() {
    // Returns string type for a service object
    return 'bair_car/sim_envRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '607013531ceb4010e22e1a175f27b233';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    float32 steer
    float32 motor
    float32 vel
    bool reset
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new sim_envRequest(null);
    if (msg.steer !== undefined) {
      resolved.steer = msg.steer;
    }
    else {
      resolved.steer = 0.0
    }

    if (msg.motor !== undefined) {
      resolved.motor = msg.motor;
    }
    else {
      resolved.motor = 0.0
    }

    if (msg.vel !== undefined) {
      resolved.vel = msg.vel;
    }
    else {
      resolved.vel = 0.0
    }

    if (msg.reset !== undefined) {
      resolved.reset = msg.reset;
    }
    else {
      resolved.reset = false
    }

    return resolved;
    }
};

class sim_envResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.coll = null;
      this.image = null;
      this.depth = null;
      this.pose = null;
    }
    else {
      if (initObj.hasOwnProperty('coll')) {
        this.coll = initObj.coll
      }
      else {
        this.coll = false;
      }
      if (initObj.hasOwnProperty('image')) {
        this.image = initObj.image
      }
      else {
        this.image = new sensor_msgs.msg.Image();
      }
      if (initObj.hasOwnProperty('depth')) {
        this.depth = initObj.depth
      }
      else {
        this.depth = new sensor_msgs.msg.Image();
      }
      if (initObj.hasOwnProperty('pose')) {
        this.pose = initObj.pose
      }
      else {
        this.pose = new geometry_msgs.msg.Pose();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type sim_envResponse
    // Serialize message field [coll]
    bufferOffset = _serializer.bool(obj.coll, buffer, bufferOffset);
    // Serialize message field [image]
    bufferOffset = sensor_msgs.msg.Image.serialize(obj.image, buffer, bufferOffset);
    // Serialize message field [depth]
    bufferOffset = sensor_msgs.msg.Image.serialize(obj.depth, buffer, bufferOffset);
    // Serialize message field [pose]
    bufferOffset = geometry_msgs.msg.Pose.serialize(obj.pose, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type sim_envResponse
    let len;
    let data = new sim_envResponse(null);
    // Deserialize message field [coll]
    data.coll = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [image]
    data.image = sensor_msgs.msg.Image.deserialize(buffer, bufferOffset);
    // Deserialize message field [depth]
    data.depth = sensor_msgs.msg.Image.deserialize(buffer, bufferOffset);
    // Deserialize message field [pose]
    data.pose = geometry_msgs.msg.Pose.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += sensor_msgs.msg.Image.getMessageSize(object.image);
    length += sensor_msgs.msg.Image.getMessageSize(object.depth);
    return length + 57;
  }

  static datatype() {
    // Returns string type for a service object
    return 'bair_car/sim_envResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '0bb2dab39897d48a64e4bba523448fcc';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
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
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new sim_envResponse(null);
    if (msg.coll !== undefined) {
      resolved.coll = msg.coll;
    }
    else {
      resolved.coll = false
    }

    if (msg.image !== undefined) {
      resolved.image = sensor_msgs.msg.Image.Resolve(msg.image)
    }
    else {
      resolved.image = new sensor_msgs.msg.Image()
    }

    if (msg.depth !== undefined) {
      resolved.depth = sensor_msgs.msg.Image.Resolve(msg.depth)
    }
    else {
      resolved.depth = new sensor_msgs.msg.Image()
    }

    if (msg.pose !== undefined) {
      resolved.pose = geometry_msgs.msg.Pose.Resolve(msg.pose)
    }
    else {
      resolved.pose = new geometry_msgs.msg.Pose()
    }

    return resolved;
    }
};

module.exports = {
  Request: sim_envRequest,
  Response: sim_envResponse,
  md5sum() { return 'b309b81746c8edad7c966dda272a1ed4'; },
  datatype() { return 'bair_car/sim_env'; }
};
