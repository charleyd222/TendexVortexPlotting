import matplotlib.pyplot as plt
import numpy as np # type: ignore
from ctypes import *
from datetime import datetime as dt
from util import vect, vis_params, spherical_fibonacci_points, make_vis_param

#Runtime
start = dt.now()

# Load cpp
rka_iter = CDLL("./cppScripts/rka_iter").rka_iter
val_return = CDLL("./cppScripts/rka_iter").val_return
super_poynting = CDLL("./cppScripts/rka_iter").super_poynting

rka_iter.argtypes = [c_double, c_double, c_double, c_int, c_int, c_double, c_double, c_double, c_double, vis_params]
val_return.argtypes = [c_double, c_double, c_double, c_int, vis_params, c_int]
super_poynting.argtypes = [c_double, c_double, c_double, vis_params]

rka_iter.restype = vect
val_return.restype = c_double
super_poynting.restype = c_double

# Make data
gauss_dtheta = 0.001
lMax = 100
model_param, A, B = make_vis_param(lMax, f'cppScripts/csvs/power_1_omega_1/maximized_coefs_lMax_{lMax}_Power_1_power.csv')
#model_param, A, B = make_vis_param(lMax, f'cppScripts/csvs/power_omega_1_xyz/maximized_coefs_xyz_lMax_{lMax}_Power_1_power.csv')
#model_param, A, B = make_vis_param(lMax, f'data_gauss_z_0p001_lMax_50.csv')
#model_param, A, B = make_vis_param(lMax, f'cppScripts/csvs/xyz_power_1_omega_1/maximized_coefs_xyz_lMax_{lMax}_Power_1_power.csv')

icity = 1
l = 500

# Resolution of the sphere grid
n_theta = 2*l
n_phi   = l
R   = 1.0

# Draw lines
num = 1000
seeds_r, seeds_theta, seeds_phi = spherical_fibonacci_points(num, R)
seeds = len(seeds_r)
num_its = 5
delta_0 = 10e-6
h0 = 10e-2
safety = .9
ending_tolerance = 0

phi_offset = np.pi
theta_offset = 0

theta = np.linspace(0, np.pi, n_theta) - theta_offset
phi = np.linspace(0, 2*np.pi, n_phi) - phi_offset
Phi, Theta = np.meshgrid(phi, theta)

# Color the sphere
colors = np.zeros_like(Theta)
for i in range(Theta.shape[0]):
    for j in range(Theta.shape[1]):
        th = Theta[i, j]
        ph = Phi[i, j]
        colors[i, j] = val_return(R, th, ph, icity, model_param, 1)
        #print(colors[i,j])
        #colors[i, j] = super_poynting(R, th, ph, model_param)


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot()

# Label axes
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$\theta$')
ax.set_title(r'$\ell$' +f' between 2 and {lMax} linear combination.' + r'Maximizing $F = \sqrt{F_x^2 + F_y^2 + F_z^2}$')
#ax.set_title(r'Super Kick Configuration Tendex Plot')# + r'Maximizing $F =F_z$')#= \sqrt{F_x^2 + F_y^2 + F_z^2}$')

#ax.set_xlim(0, 2 * np.pi)
#ax.set_ylim(0, np.pi)
if False:
    # Draw integral curves
    for i in range(seeds):
        r_val, theta0, phi0 = seeds_r[i], seeds_theta[i], seeds_phi[i]

        vect_c = rka_iter(r_val, theta0, phi0, num_its, icity,
                        ending_tolerance, delta_0, safety, h0, model_param)

        its = vect_c.its
        thetas = np.array(vect_c.y[0:its])
        phis   = np.array(vect_c.z[0:its])

        ax.plot(phis - phi_offset, thetas - theta_offset, color='black', linewidth=0.6, alpha=0.9)


# Heatmap uses phi horizontally and theta vertically
m = ax.imshow(
    colors,
    extent=[np.min(phi), np.max(phi), np.min(theta), np.max(theta)],
    origin='lower',
    aspect='auto',
    cmap='viridis'
)

print(dt.now() - start)
plt.colorbar(m)
plt.show()
