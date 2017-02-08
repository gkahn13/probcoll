import abc

import numpy as np

from general.utility.utils import smooth

class Noise(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, meta_data, **kwargs):
        self._udim = meta_data['U']['dim']

    @abc.abstractmethod
    def sample(self, u):
        raise NotImplementedError('Implement in subclass')


class ZeroNoise(Noise):
    def __init__(self, meta_data, **kwargs):
        Noise.__init__(self, meta_data)

    def sample(self, u):
        return np.zeros((self._udim,))

class GaussianNoise(Noise):
    def __init__(self, meta_data, **kwargs):
        Noise.__init__(self, meta_data)
        self.std = kwargs.get('std')
        if type(self.std) is float or type(self.std) is int:
            self.std = [self.std] * self._udim

    def sample(self, u):
        noise = []
        for std_i in self.std:
            if std_i > 0:
                noise.append(np.random.normal(0., std_i, 1)[0])
            else:
                noise.append(0.)
        return np.array(noise)

class UniformNoise(Noise):
    def __init__(self, meta_data, **kwargs):
        Noise.__init__(self, meta_data)
        self.lower = kwargs.get('lower')
        self.upper = kwargs.get('upper')
        assert(len(self.lower) == meta_data['U']['dim'])
        assert(len(self.upper) == meta_data['U']['dim'])

    def sample(self, u):
        return np.random.uniform(self.lower, self.upper)

class OUNoise(GaussianNoise):
    def __init__(self, meta_data, **kwargs):
        """
        sigma_t = theta * (mu - sigma_t-1) + sample noise
        """
        GaussianNoise.__init__(self, meta_data, **kwargs)
        self.theta = kwargs.get('theta')
        self.mu = np.array(kwargs.get('mu'))
        self.dt = meta_data['dt']
        assert(len(self.mu) == self._udim)

    def sample(self, u):
        return self.theta * (self.mu - u) * self.dt + GaussianNoise.sample(self, u)

class SmoothedGaussianNoise(GaussianNoise):
    def __init__(self, meta_data, **kwargs):
        GaussianNoise.__init__(self, meta_data, **kwargs)
        self.T = kwargs.get('T')
        self.noise = np.array([GaussianNoise.sample(self, None) for _ in xrange(self.T)])

    def sample(self, u):
        # smooth and return
        smoothed_noise = np.nan * np.ones(self.noise.shape)
        for i in xrange(self._udim):
            smoothed_noise[:, i] = smooth(self.noise[:, i], max(self.T//6, 3), window='hanning')
        assert(not np.isnan(smoothed_noise).all())
        sample = smoothed_noise[0, :]

        self.noise[1:,:] = self.noise[:-1,:]
        self.noise[0,:] = GaussianNoise.sample(self, u)

        return sample
