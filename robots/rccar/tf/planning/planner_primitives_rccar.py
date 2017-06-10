import tensorflow as tf
import numpy as np
from general.tf.planning.planner_primitives import PlannerPrimitives

class PlannerPrimitivesRCcar(PlannerPrimitives):
    def _create_primitives(self):
        steers = self.params['primitives']['steers']
        speeds = self.params['primitives']['speeds']
        num_splits = self.params['primitives']['num_splits']        
        controls = []
        s_len = len(steers)
        m_len = len(speeds)
        for n in xrange((s_len * m_len)**num_splits):
            s_val = n
            m_val = n // (s_len ** num_splits)
            control = []
            horizon_left = self.probcoll_model.T
            for i in xrange(num_splits):
                s_index = s_val % s_len
                s_val = s_val // s_len
                m_index = m_val % m_len
                m_val = m_val // m_len
                cur_len = horizon_left // (num_splits - i)
                control += [[steers[s_index], speeds[m_index]]] * cur_len
                horizon_left -= cur_len
            controls.append(np.array(control))
        controls = np.array(controls)
        self.primitives = tf.constant(controls, dtype=self.probcoll_model.dtype)
