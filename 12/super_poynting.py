import matplotlib.pyplot as plt
import numpy as np
from ctypes import *
from datetime import datetime as dt
from util import vect, vis_params, spherical_fibonacci_points
from scipy.integrate import simpson

#Runtime
start = dt.now()

super_poynting = CDLL("./cppScripts/rka_iter").super_poynting

super_poynting.argtypes = [c_double, c_double, c_double, vis_params]
super_poynting.restype = c_double

model_param = vis_params(
    M=1.,
    omega1=1,
    omega2=1,
    t=0.,
    C2=.1,
    w2=np.pi/4,
    ell=30
)
R = 1

n_phi = 200
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
plt.title('Super-Poynting Vector integrated in each grid point. Reconstructed E0 and B0, for l <= 30')
print(dt.now() - start)
plt.colorbar()
plt.show()