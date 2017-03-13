import numpy as np

from general.dynamics.dynamics import Dynamics

class DynamicsBebop2d(Dynamics):

    def __init__(self):
        Dynamics.__init__(self)

        # self.A = np.array([[0., 0.],
        #                    [0., 0.]])
        #
        # self.B = np.array([[1., 0.],
        #                    [0., 1.]])
        self.A = np.zeros([3,3])
        self.B = np.eye(3)

    def evolve(self, x, u, fx=None, fu=None, f0=None):
        """
        Evolve the state x given control u, return x_next = x + dt * (dx/dt)
        """
        x_tp1 = self.A.dot(x) + self.B.dot(u)
        return x_tp1

    def _derivative(self, x, u):
        """
        Give the time derivative, i.e. evaluate the equation of motion
        Return x_dot = dx/dt = F(x, u).
        """
        dAdt = np.array([[0., 0.],
                         [0., 0.]])

        dBdt = np.array([[0., 0.],
                         [0., 0.]])

        return dAdt.dot(x) + dBdt.dot(u)

    def _jacobian(self, x, u):
        """
        Give the dX by dX+dU Jacobian matrix of F: (x, u) -> dx
        """
        return np.copy(self.A), np.copy(self.B)
