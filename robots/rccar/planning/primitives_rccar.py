import numpy as np

from general.planning.primitives.primitives import Primitives
from general.state_info.sample import Sample

from config import params

class PrimitivesRCcar(Primitives):

    def _create_primitives(self):
        des_vel = params['planning']['cost_velocity']['u_des']
        weights = params['planning']['cost_velocity']['u_weights']

        steers = [30., 40., 50., 60., 70.]
        speeds = [8., 12., 16.]

        samples = []
        for steer1 in steers:
            for steer2 in steers:
                for steer3 in steers:
                    speed1 = 16.
                    sample = Sample(T=self._H)
                    linearvel1 = [steer1, speed1]
                    linearvel2 = [steer2, speed1]
                    linearvel3 = [steer3, speed1]
                    sample.set_U(linearvel1, t=slice(0, self._H//3))
                    sample.set_U(linearvel2, t=slice(self._H//3, (self._H*2)//3))
                    sample.set_U(linearvel3, t=slice((self._H*2)//3, self._H))
                    samples.append(sample)

#        for steer1 in steers:
#            sample = Sample(T=self._H)
#            linearvel1 = [steer1, 20.]
#            sample.set_U(linearvel1, t=slice(0, self._H))
#            samples.append(sample)
        return samples
