import numpy as np

from general.planning.primitives.primitives import Primitives
from general.state_info.sample import Sample

from config import params

class PrimitivesBebop2d(Primitives):

    def _create_primitives(self):
        des_vel = params['planning']['cost_velocity']['velocity']
        weights = params['planning']['cost_velocity']['weights']

        assert(weights[1] == 0 and weights[2] == 0)

        thetas = np.linspace(-np.pi/2., np.pi/2., 19)
        speeds = np.linspace(0.1, 1.0, 10) * des_vel[0]

        samples = []
        for theta in thetas:
            for speed in speeds:
                sample = Sample(T=self._H)
                linearvel = [speed * np.cos(theta), speed * np.sin(theta), 0.]
                sample.set_U(linearvel, t=slice(0, self._H))
                samples.append(sample)

        return samples