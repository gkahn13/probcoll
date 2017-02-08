import cython
cimport numpy as np
from libcpp cimport bool

cdef extern from "c_ilqr_backward.h":
    void c_ilqr_backward(int T, int xdim, int udim, double mu, bool reg_state, bool reg_control,
             double *Fx, double *Fu,
             double *lu, double *lx, double *luu, double *lxx, double *lux,
             double *dV, double *Vx, double *Vxx,
             double *Qx, double *Qu, double *Qxx, double *Quu, double *Qux,
             double *k, double *K, double *inv_pol_covar, double *pol_covar, double *chol_pol_covar)

@cython.boundscheck(False)
@cython.wraparound(False)
def ilqr_backward(int T, int xdim, int udim, double mu, bool reg_state, bool reg_control,
             np.ndarray[double, ndim=3, mode="c"] Fx not None,
             np.ndarray[double, ndim=3, mode="c"] Fu not None,
             np.ndarray[double, ndim=2, mode="c"] lu not None,
             np.ndarray[double, ndim=2, mode="c"] lx not None,
             np.ndarray[double, ndim=3, mode="c"] luu not None,
             np.ndarray[double, ndim=3, mode="c"] lxx not None,
             np.ndarray[double, ndim=3, mode="c"] lux not None,
             np.ndarray[double, ndim=2, mode="c"] dV not None,
             np.ndarray[double, ndim=2, mode="c"] Vx not None,
             np.ndarray[double, ndim=3, mode="c"] Vxx not None,
             np.ndarray[double, ndim=2, mode="c"] Qx not None,
             np.ndarray[double, ndim=2, mode="c"] Qu not None,
             np.ndarray[double, ndim=3, mode="c"] Qxx not None,
             np.ndarray[double, ndim=3, mode="c"] Quu not None,
             np.ndarray[double, ndim=3, mode="c"] Qux not None,
             np.ndarray[double, ndim=2, mode="c"] k not None,
             np.ndarray[double, ndim=3, mode="c"] K not None,
             np.ndarray[double, ndim=3, mode="c"] inv_pol_covar not None,
             np.ndarray[double, ndim=3, mode="c"] pol_covar not None,
             np.ndarray[double, ndim=3, mode="c"] chol_pol_covar not None):

    c_ilqr_backward(T, xdim, udim, mu, reg_state, reg_control,
                   &Fx[0,0,0], &Fu[0,0,0],
                   &lu[0,0], &lx[0,0], &luu[0,0,0], &lxx[0,0,0], &lux[0,0,0],
                   &dV[0,0], &Vx[0,0], &Vxx[0,0,0],
                   &Qx[0,0], &Qu[0,0], &Qxx[0,0,0], &Quu[0,0,0], &Qux[0,0,0],
                   &k[0,0], &K[0,0,0], &inv_pol_covar[0,0,0], &pol_covar[0,0,0], &chol_pol_covar[0,0,0])

    return None
