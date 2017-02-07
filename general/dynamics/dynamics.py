import abc
import numpy as np

from config import params

class Dynamics(object):
    """
    Linearized approximation of dynamics
    """

    def __init__(self):
        self.dt = params['dt']
        self.dX = params['X']['dim']
        self.dU = params['U']['dim']

    @abc.abstractmethod
    def evolve(self, x, u, fx=None, fu=None, f0=None):
        """
        Evolve the state x given control u, return x_next = x + dt * (dx/dt)
        """
        raise NotImplementedError("Must be implemented in subclass")

    def linearize_batch(self, traj):
        """
        Linearize this dynamics around the given trajectory

        :type traj: Sample
        :rtype: (np.ndarray, np.ndarray, np.ndarray)
        """
        T = traj.T
        Fx = np.zeros((T, self.dX, self.dX))
        Fu = np.zeros((T, self.dX, self.dU))
        fv = np.zeros([T, self.dX])  # zero order approx

        for t in range(T):
            x, u = traj.get_X(t=t), traj.get_U(t=t)  # check dimension
            fx, fu, f0 = self.linearize(x, u)
            Fx[t] = fx
            Fu[t] = fu
            fv[t] = f0
        return Fx, Fu, fv

    def linearize(self, x_ref, u_ref):
        """
        Return the linearized dynamics around the reference point
        such that x_next = fx * x + fu * u + f0

        Let F(x, u) be the real dynamics function s.t. dx = F(x, u).
        Approx around ref: dx = F(x_ref, u_ref) + Jx * (x - x_ref) + Ju * (u - u_ref)
        Discretize: x_next = x + dt * dx, that is
        f0 = dt * ( F(x_ref, u_ref) - Jx * x_ref - Ju * u_ref )
        fx = I + dt * Jx, fu = dt * Ju
        """
        # derivatives/jacobians
        dxdt = self._derivative(x_ref, u_ref)
        Jx, Ju = self._jacobian(x_ref, u_ref)

        fu = self.dt * Ju
        fx = np.eye(self.dX) + self.dt * Jx
        f0 = self.dt * (dxdt - Jx.dot(x_ref) - Ju.dot(u_ref))

        return f0, fx, fu

    @abc.abstractmethod
    def _derivative(self, x, u):
        """
        :return: dxdt(x,u)
        """
        raise NotImplementedError("Must be implemented in subclass")

    @abc.abstractmethod
    def _jacobian(self, x, u):
        """
        :return: Jx(x,u), Ju(x,u)
        """
        raise NotImplementedError("Must be implemented in subclass")
