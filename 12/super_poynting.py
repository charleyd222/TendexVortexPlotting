import matplotlib.pyplot as plt
import numpy as np
from ctypes import *
from datetime import datetime as dt
from util import vect, vis_params, make_vis_param
from scipy.integrate import simpson

#Runtime
start = dt.now()

super_poynting = CDLL("./cppScripts/rka_iter").super_poynting

super_poynting.argtypes = [c_double, c_double, c_double, vis_params]
super_poynting.restype = c_double

lMax=24
model_param, A, B = make_vis_param(lMax, f'cppScripts/csvs/xyz_power_1_omega_1/maximized_coefs_xyz_lMax_{lMax}_Power_1_power.csv')
#model_param, A, B = make_vis_param(lMax, f'cppScripts/csvs/power_1_omega_1/maximized_coefs_lMax_{lMax}_Power_1_power.csv')

R = 1

n_phi = 100
n_theta = n_phi//2

phi = np.linspace(0,2*np.pi, n_phi)
theta = np.linspace(0, np.pi, n_theta)

dphi = phi[1] - phi[0]
dtheta = theta[1] - theta[0]

P = np.zeros((n_phi, n_theta))

for i, p in enumerate(phi):
    for j, t in enumerate(theta):
        P[i, j] = super_poynting(R, t, p, model_param)

local_flux = P * R**2 * np.sin(theta)[None, :] * dtheta * dphi

plt.imshow(
    local_flux.T,
    extent=[0, 2*np.pi, 0, np.pi],
    origin='lower',
    aspect='auto',
    cmap='viridis'
)

plt.xlabel('Phi')
plt.ylabel('Theta')
plt.title(f'Super-Poynting Vector integrated in each grid point.\n Reconstructed E0 and B0, for $\ell$ <= {lMax}')
print(dt.now() - start)
plt.colorbar()
plt.show()