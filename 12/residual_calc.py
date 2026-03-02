import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
from ctypes import *
from datetime import datetime as dt
from util import vis_params, mat, spherical_fibonacci_points, make_vis_param

#Runtime
start = dt.now()

# Load cpp
mat_return_E = CDLL("./cppScripts/rka_iter").mat_return
mat_return_E0 = CDLL("./cppScripts/rka_iter_E0").mat_return

mat_return_E.argtypes = [c_double, c_double, c_double, vis_params]
mat_return_E0.argtypes = [c_double, c_double, c_double, vis_params]

mat_return_E.restype = mat
mat_return_E0.restype = mat

# Make data
gauss_dtheta = 0.01
coef_length = 50
lMax = 50
model_param, A, B = make_vis_param(gauss_dtheta, coef_length, lMax)

icity = 1
l = 500

# Resolution of the sphere grid
n_theta = 2*l
n_phi   = l
R   = 1.0

theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2*np.pi, n_phi)
Phi, Theta = np.meshgrid(phi, theta)

# Color the sphere
colors = np.zeros_like(Theta)
E0 = np.zeros((Theta.shape[0], Theta.shape[1], 2, 2))
for i in range(Theta.shape[0]):
        for j in range(Theta.shape[1]):
            th = Theta[i, j]
            ph = Phi[i, j]

            E0_T = mat_return_E0(R, th, ph, model_param)
            E0_T = np.array([[E0_T.m11, E0_T.m12], [E0_T.m21, E0_T.m22]])
            E0[i, j, :, :] = E0_T

res = []
for l in range(2,50):
    resT = []
    for i in range(Theta.shape[0]):
        for j in range(Theta.shape[1]):
            th = Theta[i, j]
            ph = Phi[i, j]
            model_param.ell = l
            E = mat_return_E(R, th, ph, model_param)
            E = np.array([[E.m11, E.m12], [E.m21, E.m22]])
            
            resT += [E - E0[i, j]]
    print(l, np.mean(resT))

    res += [np.mean(resT)]

plt.plot(res, np.linspace(2,50,len(res)))
plt.show()
