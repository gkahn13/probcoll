import numpy as np
from scipy.linalg import cholesky, solve_triangular, LinAlgError

from general.planning.cost.approx import ValueApprox, LocalValueApprox, DynamicsApprox, CostApprox
from general.planning.ilqr.util.info import iLQRInfo
from general.policy.lin_gauss_policy import LinearGaussianPolicy
from general.policy.noise_models import ZeroNoise
from general.utility.logger import get_logger

from config import params as meta_data

class iLQR(object):
    def __init__(self, config=None):
        self._config = config if config is not None else meta_data['ilqr']
        self._logger = get_logger(self.__class__.__name__, 'fatal')

    def plan(self, init_traj, dynamics, cost_func, init_policy=None, plot_env=None, **kwargs):
        """
        :type init_traj: Sample
        :type dynamics: Dynamics
        :type cost_func: Cost
        :type init_policy: LinearGaussianPolicy
        """
        T = init_traj._T
        if init_policy is None:
            init_policy = LinearGaussianPolicy(T, init=True, meta_data=kwargs.get('meta_data', meta_data))

        success, new_traj, new_policy, new_cst, new_dyn, new_V, new_Q = \
            self._ilqr(init_traj, dynamics, cost_func, init_policy, plot_env=plot_env, **kwargs)

        new_traj_info = iLQRInfo(
            traj=new_traj,
            lqr_policy=new_policy,
            V_approx=new_V,
            Q_approx=new_Q,
            dyn_approx=new_dyn,
            cst_approx=new_cst,
            cost_func=cost_func,
            dynamics=dynamics
        )

        return success, new_traj_info

    def _ilqr(self, init_traj, dynamics, cost_func, init_policy, plot_env=None, **kwargs):
        T = init_traj._T
        self._init_params()
        curr_traj = curr_policy = curr_cst = curr_dyn = V = Q = None
        ilqr_success = True

        self._logger.info('=== iLQR: time horizon {0}'.format(T))

        # if 'curr_traj' in kwargs and 'curr_dyn' in kwargs and 'curr_cst' in kwargs:
        #     success = True
        #     curr_traj = kwargs.pop('curr_traj')
        #     curr_dyn = kwargs.pop('curr_dyn')
        #     curr_cst = kwargs.pop('curr_cst')
        # else:
        success, curr_traj, curr_dyn, curr_cst = \
            self._forward(init_traj, init_policy, dynamics, cost_func,
                          use_traj_u=kwargs.pop('use_traj_u', True))

        if not success:
            ilqr_success = False
            self._logger.error("Diverge in the first forward pass")

        if plot_env:
            plot_env.rave_env.clear_plots()
            curr_traj.plot_rave(plot_env.rave_env, color=(0,0,1))
            raw_input('Press enter')

        # iLQR main loop
        while ilqr_success and self._curr_iter < min(self._max_iter, kwargs.get('max_iter', np.inf)):
            self._curr_iter += 1
            self._logger.info('i=%2d: cost= %f, final state=%s',
                             self._curr_iter, curr_cst.J, curr_traj.get_X(t=-1))

            # backward pass
            back_success = False
            new_policy = init_policy.empty_like()
            while not back_success:
                self._logger.info('\tBackpass with mu=%f', self.mu)
                back_success, dV, new_V, new_Q = self._backward(
                    curr_traj, curr_dyn, curr_cst, new_policy, **kwargs
                )
                if not back_success and not self._increase_mu(): break
            if not back_success:
                # self._logger.error('=== iLQR fails due to backpass')
                # ilqr_success = False
                self._logger.warn('=== (backward pass) iLQR breaking because mu reached mu max')
                ilqr_success = True
                break
            else:
                V = new_V
                Q = new_Q
                curr_policy = new_policy

            # line search to find next trajectory and cost/dynamics approx
            line_search_done = False
            alpha = self._config['alpha_start']
            best_k, best_traj, best_dyn, best_cst = None, None, None, None
            z_best = -np.inf
            while alpha > self._config['alpha_min']:
                curr_policy.k = alpha * curr_policy.k
                success, next_traj, next_dyn, next_cst = \
                    self._forward(curr_traj, curr_policy, dynamics, cost_func)
                if success:
                    cost_delta = curr_cst.J - next_cst.J
                    cost_delta_ratio = cost_delta / curr_cst.J
                    expected = -alpha * (dV[0] + alpha * dV[1])
                    z = cost_delta / expected
                    self._logger.info(
                        '\tcost->%f (%f), z=%f, alpha=%f',
                        next_cst.J, cost_delta_ratio, z, alpha
                    )

                    if z > z_best:
                        z_best = z
                        best_k = np.copy(curr_policy.k)
                        best_traj, best_dyn, best_cst = next_traj, next_dyn, next_cst
                else:
                    expected = np.inf
                    z = -np.inf
                    self._logger.info('\tforward fails, alpha=%f', alpha)

                if expected < 0:
                    self._logger.error('Negative expected cost reduction')
                    z = np.sign(cost_delta)
                if z > self._config['z_min']:  # satisfied cost reduction
                    line_search_done = True
                    curr_traj, curr_dyn, curr_cst = next_traj, next_dyn, next_cst
                    self._decrease_mu()
                    break  # break the line search loop
                # decrease alpha
                alpha *= self._config['alpha_mult']
            # end of line search

            ### TODO: line search: take best alpha (only if alpha is going to 0)
            if not line_search_done: # failed, restart current traj
                self._logger.warn('\tLine search failed: mu=%f', self.mu)
                self._logger.warn('\tTaking z_best: %f', z_best)
                curr_policy.k = best_k
                curr_traj, curr_dyn, curr_cst = best_traj, best_dyn, best_cst

                if not self._increase_mu():
                    # self._logger.error('===iLQR fails due to line search')
                    # ilqr_success = False
                    self._logger.warn('=== (line search) iLQR breaking because mu reached mu max')
                    ilqr_success = True
                    break
            else:
                # check for termination
                if cost_delta_ratio < self._config['min_cost_delta']:
                    self._logger.info('Terminate: small cost reduction ratio')
                    break  # break the main loop

            # # line search to find next trajectory and cost/dynamics approx
            # line_search_done = False
            # alpha = self._config['alpha_start']
            # while alpha > self._config['alpha_min']:
            #     curr_policy.k = alpha * curr_policy.k
            #     success, next_traj, next_dyn, next_cst = \
            #         self._forward(curr_traj, curr_policy, dynamics, cost_func)
            #     if success:
            #         cost_delta = curr_cst.J - next_cst.J
            #         cost_delta_ratio = cost_delta / curr_cst.J
            #         expected = -alpha * (dV[0] + alpha * dV[1])
            #         z = cost_delta / expected
            #         self._logger.info(
            #             '\tcost->%f (%f), z=%f, alpha=%f',
            #             next_cst.J, cost_delta_ratio, z, alpha
            #         )
            #     else:
            #         expected = np.inf
            #         z = -np.inf
            #         self._logger.info('\tforward fails, alpha=%f', alpha)
            #
            #     if expected < 0:
            #         self._logger.error('Negative expected cost reduction')
            #         z = np.sign(cost_delta)
            #     if z > self._config['z_min']:  # satisfied cost reduction
            #         line_search_done = True
            #         curr_traj, curr_dyn, curr_cst = next_traj, next_dyn, next_cst
            #         self._decrease_mu()
            #         break  # break the line search loop
            #     # decrease alpha
            #     alpha *= self._config['alpha_mult']
            # # end of line search
            #
            # if not line_search_done: # failed, restart current traj
            #     self._logger.warn('\tLine search failed: mu=%f', self.mu)
            #     if not self._increase_mu():
            #         # self._logger.error('===iLQR fails due to line search')
            #         # ilqr_success = False
            #         self._logger.warn('=== (line search) iLQR breaking because mu reached mu max')
            #         ilqr_success = True
            #         break
            # else:
            #     # check for termination
            #     if cost_delta_ratio < self._config['min_cost_delta']:
            #         self._logger.info('Terminate: small cost reduction ratio')
            #         break  # break the main loop

            if plot_env:
                plot_env.rave_env.clear_plots()
                curr_traj.plot_rave(plot_env.rave_env, color=(0,0,1))
                raw_input('Press enter')

        if not np.isfinite(curr_cst.J):
            self._logger.error("=== iLQR fails: infinite final cost")
            ilqr_success = False
        return ilqr_success, curr_traj, curr_policy, curr_cst, curr_dyn, V, Q

    def _init_params(self):
        # current iLQR iteration
        self._curr_iter = 0
        # regularization parameters
        self.mu = self._config['mu_start']
        self.dmu = self._config['dmu_start']
        # maximum number of iLQR iteration
        self._max_iter = self._config['max_iter']

    def _forward(self, traj, policy, dynamics, cost_func, use_traj_u=False):
        """
        Execute policy with reference trajectory
        Calculate approximation to dynamics and cost along trajectory
        This method must not modify any arguments

        :type traj: Sample
        :type policy: LinearGaussianPolicy
        :type dynamics: Dynamics
        :type cost_func: Cost
        :rtype: (Sample, DynamicsApprox, CostApprox)
        """
        assert(np.isfinite(traj.get_U()).all())
        assert(np.isfinite(traj.get_X()).all())

        T = traj._T
        xdim = traj._xdim
        udim = traj._udim
        new_traj = traj.empty_like()
        dyn = DynamicsApprox(T, xdim, udim)
        zero_noise = ZeroNoise(policy._meta_data)

        new_traj.set_O(traj.get_O(t=0), t=0)
        x = traj.get_X(t=0)
        for t in range(T):
            if np.amax(abs(x)) > 1e8:
                self._logger.warn('Forward pass diverges: t=%d, x=%s', t, x)
                cst = CostApprox(T, xdim, udim)
                cst.J = np.inf
                return False, traj, None, cst
            # set (x, u)
            new_traj.set_X(x, t=t)
            if use_traj_u:
                u = traj.get_U(t=t)
            else:
                u = policy.act(x, 'no_obs', t, ref_traj=traj, noise=zero_noise)
            new_traj.set_U(u, t=t)
            # linearize
            f0, fx, fu = dynamics.linearize(x, u)
            dyn.set_t(t, Fx=fx, Fu=fu, f0=f0)
            # evolve to next x_tp1
            x = dynamics.evolve(x, u, fx=fx, fu=fu, f0=f0)

        cst = cost_func.eval(new_traj)

        return True, new_traj, dyn, cst

    def _backward(self, traj, d, c, policy, **kwargs):
        """
        Calculate a quadratic approximation of value for all t
        Cost-to-go: J_i(x, U_i) = l_f(x_T) + sum_{t=i}^{T-1} l(x_t, u_t)
        Value: V(x, i) = min_{U_i} J_i(x, U_i)
                       = min_u l(x, u) + V(f(x, u), i+1)
        This method should only modify argument policy

        :type: traj: Sample
        :type: d: DynamicsApprox
        :type: c: CostApprox
        :type policy: LinearGaussianPolicy
        :rtype: ValueApprox
        """
        T = traj._T
        xdim = traj._xdim
        udim = traj._udim

        Fx = np.ascontiguousarray(d.Fx)
        Fu = np.ascontiguousarray(d.Fu)
        lu = np.ascontiguousarray(c.lu)
        lx = np.ascontiguousarray(c.lx)
        luu = np.ascontiguousarray(c.luu)
        lxx = np.ascontiguousarray(c.lxx)
        lux = np.ascontiguousarray(c.lux)

        dV = np.ascontiguousarray(np.zeros((T, 2)))
        Vx = np.ascontiguousarray(np.zeros((T, xdim)))
        Vxx = np.ascontiguousarray(np.zeros((T, xdim, xdim)))
        Qx = np.ascontiguousarray(np.zeros((T, xdim)))
        Qu = np.ascontiguousarray(np.zeros((T, udim)))
        Qxx = np.ascontiguousarray(np.zeros((T, xdim, xdim)))
        Quu = np.ascontiguousarray(np.zeros((T, udim, udim)))
        Qux = np.ascontiguousarray(np.zeros((T, udim, xdim)))
        k = np.ascontiguousarray(np.zeros((T, udim)))
        K = np.ascontiguousarray(np.zeros((T, udim, xdim)))
        inv_pol_covar = np.ascontiguousarray(np.zeros((T, udim, udim)))
        pol_covar = np.ascontiguousarray(np.zeros((T, udim, udim)))
        chol_pol_covar = np.ascontiguousarray(np.zeros((T, udim, udim)))

        from cython.ilqr_backward import ilqr_backward
        ilqr_backward(T, xdim, udim,
                      self.mu, self._config['reg_state'], self._config['reg_control'],
                      Fx, Fu,
                      lu, lx, luu, lxx, lux,
                      dV, Vx, Vxx,
                      Qx, Qu, Qxx, Quu, Qux,
                      k, K, inv_pol_covar, pol_covar, chol_pol_covar)

        success = True

        policy.k = k
        policy.K = K
        policy.inv_pol_covar = inv_pol_covar
        policy.pol_covar = pol_covar
        policy.chol_pol_covar = chol_pol_covar

        V = ValueApprox(T, xdim)
        V.dV, V.Vx, V.Vxx = dV, Vx, Vxx

        Q = LocalValueApprox(T, xdim, udim)
        Q.Qx, Q.Qu, Q.Qxx, Q.Quu, Q.Qux = Qx, Qu, Qxx, Quu, Qux

        return success, dV.sum(axis=0), V, Q

        # success = True
        # T = traj._T
        # V = ValueApprox(T, traj._xdim)
        # Q = LocalValueApprox(T, traj._xdim, traj._udim)
        # # last step dynamics not used, last step cost used
        # dV = kwargs.get('dV', np.zeros((2,)))
        # if dV is None: dV = np.zeros((2,))
        # Vx = kwargs.get('Vx', None)
        # Vxx = kwargs.get('Vxx', None)
        # zero_out_k = kwargs.get('zero_out_k', False)
        # for t in range(T-1, -1, -1):
        #     # below are pointers, make sure their reference are not modified
        #     Qu, Qx = c.lu[t], c.lx[t]
        #     Quu, Qxx, Qux = c.luu[t], c.lxx[t], c.lux[t]
        #     Quu_reg, Qux_reg = c.luu[t], c.lux[t]
        #     # next step value if not final, or warm start provided
        #     if Vx is not None and Vxx is not None:
        #         Qu = Qu + d.Fu[t].T.dot(Vx)
        #         Qx = Qx + d.Fx[t].T.dot(Vx)
        #         Qxx = Qxx + d.Fx[t].T.dot(Vxx.dot(d.Fx[t]))
        #         Quu = Quu + d.Fu[t].T.dot(Vxx.dot(d.Fu[t]))
        #         Qux = Qux + d.Fu[t].T.dot(Vxx.dot(d.Fx[t]))
        #         Quu_reg, Qux_reg = np.copy(Quu), np.copy(Qux)
        #         if self._config['reg_state']:
        #             Quu_reg += self.mu * d.Fu[t].T.dot(d.Fu[t])
        #             Qux_reg += self.mu * d.Fu[t].T.dot(d.Fx[t])
        #         if self._config['reg_control']:
        #             Quu_reg += self.mu * np.eye(traj._udim)
        #     try:
        #         L = cholesky(Quu_reg, lower=True)
        #     except LinAlgError as e:
        #         eigvals = np.linalg.eigvals(Quu_reg)
        #         self._logger.debug('Failed Cholesky at t=%d: %s', t, e.message)
        #         self._logger.debug('Quu_reg (eig: %s) = \n%s', eigvals, Quu_reg)
        #         success = False
        #         break
        #     # calculate control gains. combine Qu and Qux for performance
        #     rhs = np.column_stack((Qu, Qux_reg))  # right-hand side
        #     kK = np.linalg.solve(L.T, np.linalg.solve(L, rhs)) # kK = solve_triangular(L.T, solve_triangular(L, rhs, lower=True))
        #     k, K = -kK[:, 0], -kK[:, 1:]
        #     if zero_out_k: k[...] = 0.
        #     # update cost-to-go approx
        #     dV_delta = [k.T.dot(Qu), 0.5 * k.T.dot(Quu.dot(k))]
        #     dV += dV_delta
        #     Vx = Qx + K.T.dot(Quu.dot(k)) + K.T.dot(Qu) + Qux.T.dot(k)
        #     Vxx = Qxx + K.T.dot(Quu.dot(K)) + K.T.dot(Qux) + Qux.T.dot(K)
        #     Vxx = 0.5 * (Vxx + Vxx.T)
        #     # update policy
        #     policy.k[t], policy.K[t] = k, K
        #     policy.inv_pol_covar[t] = Quu_reg
        #     policy.pol_covar[t] = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(traj._udim)))
        #     # policy.pol_covar[t] = solve_triangular(
        #     #     L.T, solve_triangular(L, np.eye(traj._udim), lower=True))
        #     policy.chol_pol_covar[t] = cholesky(policy.pol_covar[t])
        #     # log approximations
        #     V.set_t(t, dV=dV_delta, Vx=Vx, Vxx=Vxx)
        #     Q.set_t(t, Qx=Qx, Qu=Qu, Qxx=Qxx, Quu=Quu_reg, Qux=Qux_reg)
        # # end of for loop
        #
        # # # TODO: temp: save relevant variables for testing
        # # with open('backward.npz', 'w') as file:
        # #     np.savez(file,
        # #              mu=self.mu,
        # #              reg_state=self._config['reg_state'],
        # #              reg_control=self._config['reg_control'],
        # #              Fx=np.array(d.Fx),
        # #              Fu=np.array(d.Fu),
        # #              lu=np.array(c.lu),
        # #              lx=np.array(c.lx),
        # #              luu=np.array(c.luu),
        # #              lxx=np.array(c.lxx),
        # #              lux=np.array(c.lux),
        # #              dV=np.array(V.dV),
        # #              Vx=np.array(V.Vx),
        # #              Vxx=np.array(V.Vxx),
        # #              Qx=np.array(Q.Qx),
        # #              Qu=np.array(Q.Qu),
        # #              Qxx=np.array(Q.Qxx),
        # #              Quu=np.array(Q.Quu),
        # #              Qux=np.array(Q.Qux),
        # #              k=np.array(policy.k),
        # #              K=np.array(policy.K),
        # #              inv_pol_covar=np.array(policy.inv_pol_covar),
        # #              pol_covar=np.array(policy.pol_covar),
        # #              chol_pol_covar=np.array(policy.chol_pol_covar))
        #
        # return success, dV, V, Q

    def _increase_mu(self):
        self.dmu = max(self.dmu * self._config['mu_mult'],
                       self._config['mu_mult'])
        self.mu = max(self.mu * self.dmu, self._config['mu_min'])
        self._logger.debug('\t\tIncrease mu: mu->%f, dmu->%f', self.mu, self.dmu)
        if self.mu > self._config['mu_max']:
            self._logger.warn('mu=%f reaches max', self.mu)
            return False
        return True

    def _decrease_mu(self):
        self.dmu = min(self.dmu / self._config['mu_mult'],
                       1 / self._config['mu_mult'])
        self.mu = self.mu * self.dmu * (self.mu > self._config['mu_min'])
        self._logger.debug('\t\tDecrease mu: mu->%f, dmu->%f', self.mu, self.dmu)
