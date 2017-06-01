import tensorflow as tf
import numpy as np
from general.tf.planning.planner_primitives import PlannerPrimitives

class PlannerPrimitivesRCcar(PlannerPrimitives):
    def _create_primitives(self):
        steers = self.params['primitives']['steers']
        speeds = self.params['primitives']['speeds']
        num_steers = self.params['primitives']['num_steers']        
        controls = []
        s_len = len(steers)
        for n in xrange(s_len**num_steers):
            for speed in speeds:
                val = n
                control = []
                for i in xrange(num_steers):
                    index = val % s_len
                    val = val // s_len
                    control += [[steers[index], speed]] * (self.probcoll_model.T//num_steers)
                controls.append(np.array(control))
        controls = np.array(controls)
        self.primitives = tf.constant(controls, dtype=self.probcoll_model.dtype)
