import numpy as np

from general.planning.primitives.primitives import Primitives
from general.state_info.sample import Sample

from config import params

class PrimitivesRCcar(Primitives):

    def _create_primitives(self):
        des_vel = params['planning']['cost_velocity']['u_des']
        weights = params['planning']['cost_velocity']['u_weights']
        steers = params['planning']['primitives']['steers']
        speeds = params['planning']['primitives']['speeds']
        num_steers = params['planning']['primitives']['num_steers']        
        samples = []
        s_len = len(steers)
        for n in xrange(s_len**num_steers):
            for speed in speeds:
                sample = Sample(T=self._H)
                val = n
                for i in xrange(num_steers):
                    index = val % s_len
                    val = val // s_len
                    sample.set_U(
                        [steers[index], speed],
                        t=slice(
                            (i * self._H)//num_steers,
                            (((i + 1) * self._H)//num_steers)))
                samples.append(sample)
        return samples
