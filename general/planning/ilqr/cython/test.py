import numpy as np

from ilqr_backward import ilqr_backward

npz = np.load('backward.npz')

# T = 2
# xdim = 2
# udim = 2
# mu = 1.0
# reg_state = True
# reg_control = True
# Fx = np.ascontiguousarray(np.zeros((T, xdim, xdim)))
# Fu = np.ascontiguousarray(np.zeros((T, xdim, udim)))
# lu = np.ascontiguousarray(np.zeros((T, udim)))
# lx = np.ascontiguousarray(np.zeros((T, xdim)))
# luu = np.ascontiguousarray(np.zeros((T, udim, udim)))
# lxx = np.ascontiguousarray(np.zeros((T, xdim, xdim)))
# lux = np.ascontiguousarray(np.zeros((T, udim, xdim)))
# dV = np.ascontiguousarray(np.zeros((T, 2)))
# Vx = np.ascontiguousarray(np.zeros((T, xdim)))
# Vxx = np.ascontiguousarray(np.zeros((T, xdim, xdim)))
# Qx = np.ascontiguousarray(np.zeros((T, xdim)))
# Qu = np.ascontiguousarray(np.zeros((T, udim)))
# Qxx = np.ascontiguousarray(np.zeros((T, xdim, xdim)))
# Quu = np.ascontiguousarray(np.zeros((T, udim, udim)))
# Qux = np.ascontiguousarray(np.zeros((T, udim, xdim)))

mu = float(npz['mu'])
reg_state = bool(npz['reg_state'])
reg_control = bool(npz['reg_control'])
Fx = np.ascontiguousarray(npz['Fx'])
Fu = np.ascontiguousarray(npz['Fu'])
lu = np.ascontiguousarray(npz['lu'])
lx = np.ascontiguousarray(npz['lx'])
luu = np.ascontiguousarray(npz['luu'])
lxx = np.ascontiguousarray(npz['lxx'])
lux = np.ascontiguousarray(npz['lux'])
T = Fx.shape[0]
xdim = Fx.shape[-1]
udim = Fu.shape[-1]

gt_dV = np.ascontiguousarray(npz['dV'])
gt_Vx = np.ascontiguousarray(npz['Vx'])
gt_Vxx = np.ascontiguousarray(npz['Vxx'])
gt_Qx = np.ascontiguousarray(npz['Qx'])
gt_Qu = np.ascontiguousarray(npz['Qu'])
gt_Qxx = np.ascontiguousarray(npz['Qxx'])
gt_Quu = np.ascontiguousarray(npz['Quu'])
gt_Qux = np.ascontiguousarray(npz['Qux'])
gt_k = np.ascontiguousarray(npz['k'])
gt_K = np.ascontiguousarray(npz['K'])
gt_inv_pol_covar = np.ascontiguousarray(npz['inv_pol_covar'])
gt_pol_covar = np.ascontiguousarray(npz['pol_covar'])
gt_chol_pol_covar = np.ascontiguousarray(npz['chol_pol_covar'])


dV = np.ascontiguousarray(np.zeros(gt_dV.shape))
Vx = np.ascontiguousarray(np.zeros(gt_Vx.shape))
Vxx = np.ascontiguousarray(np.zeros(gt_Vxx.shape))
Qx = np.ascontiguousarray(np.zeros(gt_Qx.shape))
Qu = np.ascontiguousarray(np.zeros(gt_Qu.shape))
Qxx = np.ascontiguousarray(np.zeros(gt_Qxx.shape))
Quu = np.ascontiguousarray(np.zeros(gt_Quu.shape))
Qux = np.ascontiguousarray(np.zeros(gt_Qux.shape))
k = np.ascontiguousarray(np.zeros(gt_k.shape))
K = np.ascontiguousarray(np.zeros(gt_K.shape))
inv_pol_covar = np.ascontiguousarray(np.zeros(gt_inv_pol_covar.shape))
pol_covar = np.ascontiguousarray(np.zeros(gt_pol_covar.shape))
chol_pol_covar = np.ascontiguousarray(np.zeros(gt_chol_pol_covar.shape))

ilqr_backward(T, xdim, udim, mu, reg_state, reg_control,
              Fx, Fu,
              lu, lx, luu, lxx, lux,
              dV, Vx, Vxx,
              Qx, Qu, Qxx, Quu, Qux,
              k, K, inv_pol_covar, pol_covar, chol_pol_covar)

print('dV diff: {0}'.format(abs(gt_dV - dV).sum()))
print('Vx diff: {0}'.format(abs(gt_Vx - Vx).sum()))
print('Vxx diff: {0}'.format(abs(gt_Vxx - Vxx).sum()))
print('Qx diff: {0}'.format(abs(gt_Qx - Qx).sum()))
print('Qxx diff: {0}'.format(abs(gt_Qxx - Qxx).sum()))
print('Quu diff: {0}'.format(abs(gt_Quu - Quu).sum()))
print('Qux diff: {0}'.format(abs(gt_Qux - Qux).sum()))
print('k diff: {0}'.format(abs(gt_k - k).sum()))
print('K diff: {0}'.format(abs(gt_K - K).sum()))
print('inv_pol_covar diff: {0}'.format(abs(gt_inv_pol_covar - inv_pol_covar).sum()))
print('pol_covar diff: {0}'.format(abs(gt_pol_covar - pol_covar).sum()))
print('chol_pol_covar diff: {0}'.format(abs(gt_chol_pol_covar - chol_pol_covar).sum()))

import IPython; IPython.embed()
