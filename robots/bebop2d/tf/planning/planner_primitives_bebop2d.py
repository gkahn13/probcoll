import tensorflow as tf
import numpy as np
from general.tf.planning.planner_primitives import PlannerPrimitives

class PlannerPrimitivesBebop2d(PlannerPrimitives):
    
    def _create_primitives(self):
        des_vel = params['planning']['control_cost']['des']

        thetas = np.linspace(-np.pi/2., np.pi/2., 19)
        speeds = np.linspace(0.1, 1.0, 10) * des_vel[0]

        controls = []
        for theta in thetas:
            for speed in speeds:
                control = []
                linearvels = [speed * np.cos(theta), speed * np.sin(theta), 0.]
                control += linearvel * self.probcoll_model.T
                controls.append(np.array(control))
        controls = np.array(controls)
        self.primitives = tf.constant(controls, dtype=self.probcoll_model.dtype)
