#include <iostream>

#include "c_ilqr_backward.h"

#include <Eigen/Eigen>
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdRow;

void c_ilqr_backward(int T, int xdim, int udim, double mu, bool reg_state, bool reg_control,
             double *Fx_arr, double *Fu_arr,
             double *lu_arr, double *lx_arr, double *luu_arr, double *lxx_arr, double *lux_arr,
             double *dV_arr, double *Vx_arr, double *Vxx_arr,
             double *Qx_arr, double *Qu_arr, double *Qxx_arr, double *Quu_arr, double *Qux_arr,
             double *k_arr, double *K_arr, double *inv_pol_covar_arr, double *pol_covar_arr, double *chol_pol_covar_arr) {

    MatrixXdRow dV(2, 1);
    MatrixXdRow Vx(xdim, 1);
    MatrixXdRow Vxx(xdim, xdim);
    MatrixXdRow dV_delta(2, 1);
    dV.setZero();

    for(int t=T-1; t >=0; --t) {
        Map<MatrixXdRow> lu_t(&lu_arr[t*udim], udim, 1);
        Map<MatrixXdRow> lx_t(&lx_arr[t*xdim], xdim, 1);
        Map<MatrixXdRow> luu_t(&luu_arr[t*udim*udim], udim, udim);
        Map<MatrixXdRow> lxx_t(&lxx_arr[t*xdim*xdim], xdim, xdim);
        Map<MatrixXdRow> lux_t(&lux_arr[t*udim*xdim], udim, xdim);
        Map<MatrixXdRow> luu_reg_t(&luu_arr[t*udim*udim], udim, udim);
        Map<MatrixXdRow> lux_reg_t(&lux_arr[t*udim*xdim], udim, xdim);

        Map<MatrixXdRow> dV_t(&dV_arr[t*2], 2, 1);
        Map<MatrixXdRow> Vx_t(&Vx_arr[t*xdim], xdim, 1);
        Map<MatrixXdRow> Vxx_t(&Vxx_arr[t*xdim*xdim], xdim, xdim);

        Map<MatrixXdRow> Qu_t(&Qu_arr[t*udim], udim, 1);
        Map<MatrixXdRow> Qx_t(&Qx_arr[t*xdim], xdim, 1);
        Map<MatrixXdRow> Quu_t(&Quu_arr[t*udim*udim], udim, udim);
        Map<MatrixXdRow> Qxx_t(&Qxx_arr[t*xdim*xdim], xdim, xdim);
        Map<MatrixXdRow> Qux_t(&Qux_arr[t*udim*xdim], udim, xdim);
        MatrixXdRow Quu_reg_t(udim, udim);
        MatrixXdRow Qux_reg_t(udim, xdim);

        Map<MatrixXdRow> k_t(&k_arr[t*udim], udim, 1);
        Map<MatrixXdRow> K_t(&K_arr[t*udim*xdim], udim, xdim);
        Map<MatrixXdRow> inv_pol_covar_t(&inv_pol_covar_arr[t*udim*udim], udim, udim);
        Map<MatrixXdRow> pol_covar_t(&pol_covar_arr[t*udim*udim], udim, udim);
        Map<MatrixXdRow> chol_pol_covar_t(&chol_pol_covar_arr[t*udim*udim], udim, udim);

        if (t < T-1) {
            Map<MatrixXdRow> Fu_t(&Fu_arr[t*xdim*udim], xdim, udim);
            Map<MatrixXdRow> Fx_t(&Fx_arr[t*xdim*xdim], xdim, xdim);

            Qu_t = lu_t + Fu_t.transpose() * Vx;
            Qx_t = lx_t + Fx_t.transpose() * Vx;
            Qxx_t = lxx_t + Fx_t.transpose() * Vxx * Fx_t;
            Quu_t = luu_t + Fu_t.transpose() * Vxx * Fu_t;
            Qux_t = lux_t + Fu_t.transpose() * Vxx * Fx_t;
            Quu_reg_t = Quu_t;
            Qux_reg_t = Qux_t;

            if (reg_state) {
                Quu_reg_t += mu * Fu_t.transpose() * Fu_t;
                Qux_reg_t += mu * Fu_t.transpose() * Fx_t;
            }

            if (reg_control) {
                Quu_reg_t += mu * MatrixXd::Identity(udim, udim);
            }
        } else {
            Qu_t = lu_t;
            Qx_t = lx_t;
            Quu_t = luu_t;
            Qxx_t = lxx_t;
            Qux_t = lux_t;
            Quu_reg_t = luu_t;
            Qux_reg_t = lux_t;
        }

        MatrixXdRow L = Quu_reg_t.llt().matrixL();

        // calculate control gains. combine Qu and Qux for performance
        MatrixXdRow rhs(udim, xdim+1);
        rhs << Qu_t, Qux_reg_t;
        MatrixXdRow kK = L.transpose().lu().solve(L.lu().solve(rhs));
        MatrixXdRow k = -kK.col(0);
        MatrixXdRow K = -kK.rightCols(xdim);

        // update cost-to-go approx
        dV_delta << k.transpose() * Qu_t, 0.5 * k.transpose() * Quu_t * k;
        dV += dV_delta;
        Vx = Qx_t + K.transpose() * Quu_t * k + K.transpose() * Qu_t + Qux_t.transpose() * k;
        Vxx = Qxx_t + K.transpose() * Quu_t * K + K.transpose() * Qux_t + Qux_t.transpose() * K;
        Vxx = 0.5 * (Vxx + Vxx.transpose());

        // update policy: k, K, inv_pol_covar, pol_covar
        k_t = k;
        K_t = K;
        inv_pol_covar_t = Quu_reg_t;
        pol_covar_t = L.transpose().lu().solve(L.lu().solve(MatrixXd::Identity(udim, udim)));
        chol_pol_covar_t = pol_covar_t.llt().matrixL().transpose();

        // write down the approximations
        dV_t = dV_delta;
        Vx_t = Vx;
        Vxx_t = Vxx;
        Quu_t = Quu_reg_t;
        Qux_t = Qux_reg_t;

    }

}
