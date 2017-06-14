__all__ = ['CostApprox', 'DynamicsApprox', 'ValueApprox', 'LocalValueApprox']

import numpy as np
from general.utility.base_classes import FrozenClass


class Approx(FrozenClass):
    def __init__(self, T, xdim, udim):
        object.__setattr__(self, 'T', T)
        object.__setattr__(self, 'xdim', xdim)
        object.__setattr__(self, 'udim', udim)
        pass

    def set(self, **kwargs):
        pass

    def set_t(self, t, **kwargs):
        pass


class CostApprox(Approx):
    def __init__(self, T, xdim, udim):
        Approx.__init__(self, T, xdim, udim)
        object.__setattr__(self, 'J',   0)
        object.__setattr__(self, 'l',   np.zeros((T,)))
        object.__setattr__(self, 'lx',  np.zeros((T, xdim)))
        object.__setattr__(self, 'lu',  np.zeros((T, udim)))
        object.__setattr__(self, 'lxx', np.zeros((T, xdim, xdim)))
        object.__setattr__(self, 'luu', np.zeros((T, udim, udim)))
        object.__setattr__(self, 'lux', np.zeros((T, udim, xdim)))

    def set_t(self, t, **kwargs):
        self.l[t] = kwargs.pop('l')
        self.lx[t] = kwargs.pop('lx')
        self.lu[t] = kwargs.pop('lu')
        self.lxx[t] = kwargs.pop('lxx')
        self.luu[t] = kwargs.pop('luu')
        self.lux[t] = kwargs.pop('lux')
        self.J += np.sum(self.l[t])

    def __imul__(self, weight):
        weight = float(weight)
        if weight != 1:
            self.J *= weight
            self.l *= weight
            self.lx *= weight
            self.lu *= weight
            self.lxx *= weight
            self.luu *= weight
            self.lux *= weight
        return self

    def __iadd__(self, other):
        assert self.same_shape(other)
        self.J += other.J
        self.l += other.l
        self.lx += other.lx
        self.lu += other.lu
        self.lxx += other.lxx
        self.luu += other.luu
        self.lux += other.lux
        return self

    def same_shape(self, other):
        return \
            self.T == other.T and \
            self.xdim == other.xdim and \
            self.udim == other.udim


class DynamicsApprox(Approx):
    def __init__(self, T, xdim, udim):
        Approx.__init__(self, T, xdim, udim)
        object.__setattr__(self, 'Fx', np.zeros((T, xdim, xdim)))
        object.__setattr__(self, 'Fu', np.zeros((T, xdim, udim)))
        object.__setattr__(self, 'f0', np.zeros((T, xdim)))

    def set_t(self, t, **kwargs):
        self.Fx[t] = kwargs.pop('Fx')
        self.Fu[t] = kwargs.pop('Fu')
        self.f0[t] = kwargs.pop('f0')


class ValueApprox(Approx):
    def __init__(self, T, xdim):
        Approx.__init__(self, T, xdim, None)
        object.__setattr__(self, 'dV',  np.zeros((T, 2)))
        object.__setattr__(self, 'Vx',  np.zeros((T, xdim)))
        object.__setattr__(self, 'Vxx', np.zeros((T, xdim, xdim)))

    def set_t(self, t, **kwargs):
        self.dV[t] = kwargs.pop('dV')
        self.Vx[t] = kwargs.pop('Vx')
        self.Vxx[t] = kwargs.pop('Vxx')


class LocalValueApprox(Approx):
    def __init__(self, T, xdim, udim):
        Approx.__init__(self, T, xdim, udim)
        object.__setattr__(self, 'Qx',  np.zeros((T, xdim)))
        object.__setattr__(self, 'Qu',  np.zeros((T, udim)))
        object.__setattr__(self, 'Qxx', np.zeros((T, xdim, xdim)))
        object.__setattr__(self, 'Quu', np.zeros((T, udim, udim)))
        object.__setattr__(self, 'Qux', np.zeros((T, udim, xdim)))

    def set_t(self, t, **kwargs):
        self.Qx[t] = kwargs.pop('Qx')
        self.Qu[t] = kwargs.pop('Qu')
        self.Qxx[t] = kwargs.pop('Qxx')
        self.Quu[t] = kwargs.pop('Quu')
        self.Qux[t] = kwargs.pop('Qux')
