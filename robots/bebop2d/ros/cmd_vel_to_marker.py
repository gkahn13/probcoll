import rospy
import visualization_msgs.msg as vm
import geometry_msgs.msg as gm
import nav_msgs.msg as nm
import bebop_msgs.msg as bm

def cmd_vel_callback(msg):
    marker = vm.Marker()
    marker.header.frame_id = '/odom'
    marker.type = marker.ARROW
    marker.action = marker.ADD
    marker.points = [
        gm.Point(0, 0.7, 0),
        gm.Point(msg.linear.x, 0.7 + msg.linear.y, 0)
    ]
    marker.lifetime = rospy.Duration()
    marker.scale.x = 0.05
    marker.scale.y = 0.05
    marker.scale.z = 0.05
    marker.color.a = 1.0
    marker.color.r = 1.0

    cmd_vel_debug_pub.publish(marker)

def speed_changed_callback(msg):
    marker = vm.Marker()
    marker.header.frame_id = '/odom'
    marker.type = marker.ARROW
    marker.action = marker.ADD
    marker.points = [
        gm.Point(0, -0.7, 0),
        gm.Point(msg.speedX, -0.7 - msg.speedY, 0)
    ]
    marker.lifetime = rospy.Duration()
    marker.scale.x = 0.05
    marker.scale.y = 0.05
    marker.scale.z = 0.05
    marker.color.a = 1.0
    marker.color.b = 1.0

    speed_changed_debug_pub.publish(marker)

def odom_callback(msg):
    marker = vm.Marker()
    marker.header.frame_id = '/odom'
    marker.type = marker.ARROW
    marker.action = marker.ADD
    marker.points = [
        gm.Point(-0.3, 0, 0),
        gm.Point(-0.3 + msg.twist.twist.linear.x, msg.twist.twist.linear.y, 0)
    ]
    marker.lifetime = rospy.Duration()
    marker.scale.x = 0.05
    marker.scale.y = 0.05
    marker.scale.z = 0.05
    marker.color.a = 1.0
    marker.color.g = 1.0

    odom_debug_pub.publish(marker)
    

rospy.init_node('cmd_vel_to_marker', anonymous=True)

cmd_vel_sub = rospy.Subscriber('/vservo/cmd_vel', gm.Twist, callback=cmd_vel_callback)
cmd_vel_debug_pub = rospy.Publisher('/bebop/debug/cmd_vel', vm.Marker, queue_size=100)

speed_changed_sub = rospy.Subscriber('/bebop/states/ardrone3/PilotingState/SpeedChanged', bm.Ardrone3PilotingStateSpeedChanged, callback=speed_changed_callback)
speed_changed_debug_pub = rospy.Publisher('/bebop/debug/speed_changed', vm.Marker, queue_size=100)

#odom_sub = rospy.Subscriber('/bebop/odom', nm.Odometry, callback=odom_callback)
#odom_debug_pub = rospy.Publisher('/bebop/debug/odom', vm.Marker, queue_size=100)

rospy.spin()

