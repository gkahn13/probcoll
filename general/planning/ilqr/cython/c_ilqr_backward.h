#ifndef _C_ILQR_BACKWARD_H__
#define _C_ILQR_BACKWARD_H__

void c_ilqr_backward(int T, int xdim, int udim, double mu, bool reg_state, bool reg_control,
             double *Fx, double *Fu,
             double *lu, double *lx, double *luu, double *lxx, double *lux,
             double *dV, double *Vx, double *Vxx,
             double *Qx, double *Qu, double *Qxx, double *Quu, double *Qux,
             double *k, double *K, double *inv_pol_covar, double *pol_covar, double *chol_pol_covar);

#endif