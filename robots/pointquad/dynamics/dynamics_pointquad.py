import numpy as np

from general.dynamics.dynamics import Dynamics

from config import params

class DynamicsPointquad(Dynamics):
    """
    Linearized approximation of dynamics
    """

    def __init__(self):
        Dynamics.__init__(self)

        self.A = np.eye(3)
        self.B = self.dt * np.eye(3)

        self.x_pos_idxs = slice(params['X']['position']['idx'],
                                params['X']['position']['idx']+params['X']['position']['dim'])
        self.x_linearvel_idxs = slice(params['X']['linearvel']['idx'],
                                      params['X']['linearvel']['idx']+params['X']['linearvel']['dim'])
        self.x_ori_idxs = slice(params['X']['orientation']['idx'],
                                params['X']['orientation']['idx']+params['X']['orientation']['dim'])
        self.x_angularvel_idxs = slice(params['X']['angularvel']['idx'],
                                       params['X']['angularvel']['idx'] + params['X']['angularvel']['dim'])

    def evolve(self, x, u, fx=None, fu=None, f0=None):
        """
        Evolve the state x given control u, return x_next = x + dt * (dx/dt)
        Note that numerical errors leads to non-unit quaternion. Normalize.
        """
        assert(np.linalg.norm(x[self.x_ori_idxs] - [1.,0,0,0]) < 1e-5)
        assert(np.linalg.norm(x[self.x_angularvel_idxs]) < 1e-5)

        u = np.array(u)

        x_pos = x[self.x_pos_idxs]
        x_pos_tp1 = self.A.dot(x_pos) + self.B.dot(u)

        x_tp1 = np.copy(x)
        x_tp1[self.x_pos_idxs] = x_pos_tp1
        x_tp1[self.x_linearvel_idxs] = u

        return x_tp1

    def _derivative(self, x, u):
        """
        Give the time derivative, i.e. evaluate the equation of motion
        Return x_dot = dx/dt = F(x, u).
        """
        dAdt = np.zeros((3,3))
        dBdt = np.eye(3)

        return dAdt.dot(x) + dBdt.dot(u)

    def _jacobian(self, x, u):
        """
        Give the dX by dX+dU Jacobian matrix of F: (x, u) -> dx
        """
        return np.copy(self.A), np.copy(self.B)
