import numpy as np

import rospy
import visualization_msgs.msg as vm
import std_msgs.msg as sm
import geometry_msgs.msg as gm

from general.ros import ros_utils

rospy.init_node('cmd_to_marker', anonymous=True)

marker_pub = rospy.Publisher('/bair_car/cmd_marker', vm.Marker, queue_size=10)

cmd_steer_callback = ros_utils.RosCallbackMostRecent('/bair_car/cmd/steer',
                                                     sm.Float32,
                                                     clear_on_get=False)
cmd_vel_callback = ros_utils.RosCallbackMostRecent('/bair_car/cmd/vel',
                                                   sm.Float32,
                                                   clear_on_get=False)
rate = rospy.Rate(20.)
while not rospy.is_shutdown():
    rate.sleep()

    cmd_steer = cmd_steer_callback.get()
    cmd_vel = cmd_vel_callback.get()

    if cmd_steer is None or cmd_vel is None:
        continue

    marker = vm.Marker()
    marker.header.frame_id = '/map'
    marker.type = marker.ARROW
    marker.action = marker.ADD
    angle = (cmd_steer.data - 50.) / 100. * (np.pi / 1.5)
    speed = cmd_vel.data
    marker.points = [
        gm.Point(0., 0., 0),
        gm.Point(speed * np.cos(angle), speed * np.sin(angle), 0)
    ]
    marker.lifetime = rospy.Duration()
    marker.scale.x = 0.05
    marker.scale.y = 0.05
    marker.scale.z = 0.05
    marker.color.b = 1.0
    marker.color.a = 1.0

    marker_pub.publish(marker)
