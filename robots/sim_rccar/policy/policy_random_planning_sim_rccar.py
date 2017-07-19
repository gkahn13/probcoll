from general.policy.policy_random_planning import PolicyRandomPlanning
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
try:
    import rospy
    import visualization_msgs.msg as vm
    import geometry_msgs.msg as gm
    import robots.rccar.ros.ros_utils as ros_utils
    use_ros = True
except:
    use_ros = False

class PolicyRandomPlanningSimRCcar(PolicyRandomPlanning):
    def __init__(self, probcoll_model, params, dtype=tf.float32):
        PolicyRandomPlanning.__init__(self, probcoll_model, params, dtype=dtype)
        if use_ros and params['visualize']:
            topics = params['topics']
            self.debug_cost_probcoll_pub = ros_utils.Publisher(
                topics['debug_cost_probcoll'],
                vm.MarkerArray,
                queue_size=10)
    
    def visualize(
            self,
            actions_considered,
            action,
            action_noisy,
            coll_costs,
            control_costs):
        if use_ros:
            marker_array = vm.MarkerArray()
            T = actions_considered.shape[1]
            for i, (one_action, cost) in enumerate(zip(actions_considered, coll_costs)):
                
                origin = np.array([0., 0., 0.])
                angle = 0.
                for t in xrange(T):
                    marker = vm.Marker()
                    marker.id = i * T + t
                    marker.header.frame_id = '/map'
                    marker.type = marker.ARROW
                    marker.action = marker.ADD

                    speed, steer = one_action[t][1] / T, one_action[t][0]
                    angle += (steer - 50.) / 100. * (np.pi / 2)
                    new_origin = origin + [speed * np.cos(angle), speed * np.sin(angle), 0.]
                    marker.points = [
                        gm.Point(*origin.tolist()),
                        gm.Point(*new_origin.tolist())
                    ]
                    origin = new_origin

                    marker.lifetime = rospy.Duration()
                    marker.scale.x = 0.05
                    marker.scale.y = 0.1
                    marker.scale.z = 0.1

                    if cost == min(coll_costs):
                        rgba = (0., 1., 0., 1.)
                        marker.scale.x *= 2
                        marker.scale.y *= 2
                        marker.scale.z *= 2
                    else:
                        rgba = list(cm.hot(1 - cost/144.))
                        rgba[-1] = 0.5
                    marker.color.r, marker.color.g, marker.color.b, marker.color.a = rgba

                    marker_array.markers.append(marker)

            self.debug_cost_probcoll_pub.publish(marker_array)
