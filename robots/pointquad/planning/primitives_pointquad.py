import numpy as np

from general.planning.primitives.primitives import Primitives
from general.state_info.sample import Sample

from config import params

class PrimitivesPointquad(Primitives):

    def _create_primitives(self):
        des_vel = params['planning']['cost_velocity']['velocity']
        weights = params['planning']['cost_velocity']['weights']
        num_angles = params['planning']['primitives']['num_angles']
        num_speeds = params['planning']['primitives']['num_speeds']
        assert(weights[1] == 0 and weights[2] == 0)

        thetas = np.linspace(-np.pi/2., np.pi/2., num_angles)
        speeds = np.linspace(1.0/num_speeds, 1.0, num_speeds) * des_vel[0]

        samples = []
        for theta in thetas:
            for speed in speeds:
                sample = Sample(T=self._H)
                linearvel = [speed * np.cos(theta), speed * np.sin(theta), 0.]
                sample.set_U(linearvel, t=slice(0, self._H))
                samples.append(sample)

        return samples
