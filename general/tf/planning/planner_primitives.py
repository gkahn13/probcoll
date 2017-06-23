import tensorflow as tf
import abc
import numpy as np
from general.tf.planning.cost.cost_desired import CostDesired
from general.tf.planning.cost.cost_coll import CostColl
from general.tf.planning.planner import Planner


class PlannerPrimitives(Planner):
    __metaclass__ = abc.ABCMeta    

    def __init__(self, probcoll_model, params, dtype=tf.float32):
        self.probcoll_model = probcoll_model
        self.params = params
        self.dtype = self.probcoll_model.dtype
        self._create_primitives()
        self._setup()
        self._setup_noise()

    @abc.abstractmethod
    def _create_primitives(self):
        """
        :return tensor of primitive actions with shape (# primitives x T x dU)
        """
        raise NotImplementedError('Implement in subclass')

    def _setup(self):
        with tf.name_scope('primitives_planner'):
            with self.probcoll_model.graph.as_default():
                self.actions_considered = self.primitives
                self.O_im_input = self.probcoll_model.d_eval['O_im_input']
                self.O_vec_input = self.probcoll_model.d_eval['O_vec_input']
                stack_u = tf.concat(0, [self.primitives]*self.params['num_dp'])
                output_pred_mean, output_pred_std, output_mat_mean, output_mat_std = self.probcoll_model.graph_eval_inference(
                    stack_u,
                    O_im_input=self.O_im_input,
                    O_vec_input=self.O_vec_input,
                    reuse=True) 

                pred_mean = tf.reduce_mean(
                    tf.split(0, self.params['num_dp'], output_pred_mean), axis=0)

                pred_std = tf.reduce_mean(
                    tf.split(0, self.params['num_dp'], output_pred_std), axis=0)

                mat_mean = tf.reduce_mean(
                    tf.split(0, self.params['num_dp'], output_mat_mean), axis=0)

                mat_std = tf.reduce_mean(
                    tf.split(0, self.params['num_dp'], output_mat_std), axis=0)
                
                mat_mean = tf.reduce_mean(
                    tf.split(0, self.params['num_dp'], output_mat_mean), axis=0)
                
                control_cost_fn = CostDesired(self.params['cost']['control_cost']) 
                coll_cost_fn = CostColl(self.params['cost']['coll_cost'])
                
                self.control_costs = control_cost_fn.eval(u_samples)
                self.coll_costs = coll_cost_fn.eval(u_samples, pred_mean, mat_mean, pred_std, mat_std)

                total_cost = self.control_costs + self.coll_costs
                index = tf.cast(tf.argmin(total_costs, axis=0), tf.int32)
                self.action = self.primitives[index, 0]
