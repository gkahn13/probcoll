import numpy as np

import rospy
import visualization_msgs.msg as vm
import geometry_msgs.msg as gm

import matplotlib.cm as cm

from general.algorithm.cost_probcoll import CostProbcoll
from general.planning.cost.approx import CostApprox
from config import params

class CostProbcollRCcar(CostProbcoll):

    def __init__(self, bootstrap, **kwargs):
        CostProbcoll.__init__(self, bootstrap, **kwargs)

        rccar_topics = params['rccar']['topics']
        self.debug_cost_probcoll_pub = rospy.Publisher(rccar_topics['debug_cost_probcoll'],
                                                         vm.MarkerArray,
                                                         queue_size=10)

#    def eval_batch(self, samples):
#        min_vel = params['U']['cmd_vel']['min']
#        cst_approxes = CostProbcoll.eval_batch(self, samples,
#                speed_func=lambda s: np.linalg.norm(s.get_U(sub_control='cmd_vel') - min_vel, axis=1).mean())
#        # costs = [cst_approx.J for cst_approx in cst_approxes]
#        costs = [1. / (1. + np.exp(-(p + 0.0*s))) for p, s in zip(self.probs_mean_batch, self.probs_std_batch)]
#        cheapest_samples, cheapest_cst_approxes = zip(*sorted(zip(samples, costs),
#                                                              key=lambda x: x[1])[:])
#        self.visualize(cheapest_samples, cheapest_cst_approxes)
#        # raw_input('eval_batch press enter')
#        return cst_approxes

    def eval_batch(self, samples):
        min_vel = params['U']['cmd_vel']['min']
        speed_func=lambda s: np.linalg.norm(s.get_U(sub_control='cmd_vel') - min_vel, axis=1).mean()
        orig_samples = samples
        samples = [s for s in samples if s is not None]

        for sample in samples:
            assert(min([True]+[x_meta in sample._meta_data['X']['order'] for x_meta in self._probcoll_model.X_order]))
            assert(min([True]+[u_meta in sample._meta_data['U']['order'] for u_meta in self._probcoll_model.U_order]))
            assert(min([True]+[o_meta in sample._meta_data['O']['order'] for o_meta in self._probcoll_model.O_order]))

        T_bootstrap = self._probcoll_model.T
        T = samples[0]._T
        xdim = samples[0]._xdim
        udim = samples[0]._udim

        cst_approxs = [CostApprox(T, xdim, udim) for _ in samples]

        ### evaluate model on all time steps
        num_avg = 10 if self._probcoll_model.dropout is not None else 1  # TODO
        probs_mean_batch, probs_std_batch = self._probcoll_model.eval_sample_batch(samples,
                                                                             num_avg=num_avg,
                                                                             pre_activation=self.pre_activation)
        probs_mean_batch = np.array(probs_mean_batch).ravel()
        probs_std_batch = np.array(probs_std_batch).ravel()

        ### for recording
        self.probs_mean_batch = probs_mean_batch
        self.probs_std_batch = probs_std_batch

        for sample, cst_approx, probs_mean, probs_std in zip(samples, cst_approxs, probs_mean_batch, probs_std_batch):
            t = T_bootstrap - 1
            speed = speed_func(sample)
            sigmoid = lambda x: (1. / (1. + np.exp(-x)))

            l = eval(self.eval_cost)
            cst_approx.l[0] = l

            cst_approx.J = np.sum(cst_approx.l)
            cst_approx *= self.weight

        ### push in unfilled CostApprox for None samples
        cst_approxs_full = []
        cst_approxs_idx = 0
        for s_orig in orig_samples:
            if s_orig is not None:
                cst_approxs_full.append(cst_approxs[cst_approxs_idx])
                cst_approxs_idx += 1
            else:
                cst_inf = CostApprox(T, xdim, udim)
                cst_inf.J = np.inf
                cst_approxs_full.append(cst_inf)
        
        cst_approxes = cst_approxs_full
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

        self.debug_cost_probcoll_pub.publish(marker_array)

        # print('probs in [{0:.2f}, {1:.2f}]'.format(min(costs), max(costs)))
