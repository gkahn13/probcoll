import numpy as np

import rospy
import visualization_msgs.msg as vm
import geometry_msgs.msg as gm

import matplotlib.cm as cm

from general.algorithms.prediction.cost_prediction import CostPrediction

from config import params

class CostPredictionRCcar(CostPrediction):

    def __init__(self, bootstrap, **kwargs):
        CostPrediction.__init__(self, bootstrap, **kwargs)

        rccar_topics = params['rccar']['topics']
        self.debug_cost_prediction_pub = rospy.Publisher(rccar_topics['debug_cost_prediction'],
                                                         vm.MarkerArray,
                                                         queue_size=10)

    def eval_batch(self, samples):
        min_vel = params['U']['cmd_vel']['min']
        cst_approxes = CostPrediction.eval_batch(self, samples,
                speed_func=lambda s: np.linalg.norm(s.get_U(sub_control='cmd_vel') - min_vel, axis=1).mean())
        # costs = [cst_approx.J for cst_approx in cst_approxes]
        costs = [1. / (1. + np.exp(-(p + 0.0*s))) for p, s in zip(self.probs_mean_batch, self.probs_std_batch)]
        cheapest_samples, cheapest_cst_approxes = zip(*sorted(zip(samples, costs),
                                                              key=lambda x: x[1])[:])
        self.visualize(cheapest_samples, cheapest_cst_approxes)
        # raw_input('eval_batch press enter')
        return cst_approxes

    def visualize(self, samples, costs):
        marker_array = vm.MarkerArray()

        for i, (sample, cost) in enumerate(zip(samples, costs)):
            if sample is None:
                continue

            origin = np.array([0., 0., 0.])
            angle = 0.
            for t in xrange(len(sample.get_U())):
                marker = vm.Marker()
                marker.id = i * len(sample.get_U()) + t
                marker.header.frame_id = '/map'
                marker.type = marker.ARROW
                marker.action = marker.ADD

                speed = sample.get_U(t=t, sub_control='cmd_vel')[0] / 5.
                steer = sample.get_U(t=t, sub_control='cmd_steer')[0]
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

                if cost == min(costs):
                    rgba = (0., 1., 0., 1.)
                    marker.scale.x *= 2
                    marker.scale.y *= 2
                    marker.scale.z *= 2
                else:
                    rgba = list(cm.hot(1 - cost))
                    rgba[-1] = 0.5
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = rgba

                marker_array.markers.append(marker)

        # for id in xrange(marker.id+1, int(1e4)):
        #     marker_array.markers.append(vm.Marker(id=id, action=marker.DELETE))

        self.debug_cost_prediction_pub.publish(marker_array)

        # print('probs in [{0:.2f}, {1:.2f}]'.format(min(costs), max(costs)))