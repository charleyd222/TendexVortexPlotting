import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ctypes import *
from datetime import datetime as dt
from util import vect, rk_params, vis_params, spherical_fibonacci_points, make_vis_param, force_calc
import pandas as pd

#Runtime
start = dt.now()

# Load cpp
rka_iter = CDLL("./cppScripts/rka_iter").rka_iter
val_return_E0 = CDLL("./cppScripts/rka_iter_E0").val_return
val_return = CDLL("./cppScripts/rka_iter").val_return

rka_iter.argtypes = [c_double, c_double, c_double, c_int, c_int, c_double, c_double, c_double, c_double, vis_params]
val_return_E0.argtypes = [c_double, c_double, c_double, c_int, vis_params, c_int]
val_return.argtypes = [c_double, c_double, c_double, c_int, vis_params, c_int]

rka_iter.restype = vect
val_return_E0.restype = c_double
val_return.restype = c_double

gauss_dtheta = 0.001
lMax = 55
cmax = 0.001
coef_length = 50

model_param, A, B = make_vis_param(lMax, name='data_gauss_phi_0p001_lMax_100.csv')

# Simulation Parameters
icity = 1
R = 1.0
num_seeds = 500
n_phi = 1000 # Grid points in x
n_theta = n_phi//2 # Grid points in y

num_its = 5
delta_0 = 1e-5
h0 = 1e-1
safety = .9
ending_tolerance = .1

# Singular Find Parameters
dPhi = .01 # Singularity find radius, x
dTheta = dPhi/2 # Singularity find radius, y
edge = 0 # Edge where to find no singularity

# Grid Setup
phi_min, phi_max = 0+edge,2*np.pi-edge
theta_min, theta_max = 0+edge, np.pi-edge
theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2*np.pi, n_phi)
Phi, Theta = np.meshgrid(phi, theta)

# Sim seeds
seeds_r, seeds_theta, seeds_phi = spherical_fibonacci_points(num_seeds, R)#, phi_min = phi_min, theta_min=theta_min, phi_max = phi_max, theta_max=theta_max)
seeds = len(seeds_r)

# ------------------ Ani Setup ------------------
# Fig Setup
fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot()
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$\theta$')
ax.set_xlim(phi_min, phi_max)
ax.set_ylim(theta_min, theta_max)

colors = np.zeros_like(Theta)
E0_colors = np.zeros_like(Theta)

img = ax.imshow(
    colors,
    extent=[phi_min, phi_max, theta_min, theta_max],
    origin='lower',
    aspect='auto',
    cmap='viridis',
    vmin=0, vmax=cmax
)

scat = ax.scatter([],[])

lines = []
for _ in range(seeds):
    line, = ax.plot([], [], color='black', linewidth=0.6, alpha=0.9)
    lines.append(line)

cbar = fig.colorbar(img, ax=ax)
fig.suptitle(r'Reconstructed $E_0$ with $d\theta = %s$' % (gauss_dtheta))
cbar.set_label(r'$\lambda_+$' if icity == 1 else r'$\lambda_-$')

# ------------------ Animation update ------------------
def update(frame):
    start_frame = dt.now()
    # Animate parameter
    model_param.ell = frame

    # Color the sphere
    for i in range(Theta.shape[0]):
        for j in range(Theta.shape[1]):
            th = Theta[i, j]
            ph = Phi[i, j]
            colors[i, j] = val_return(R, th, ph, icity, model_param, 1)
            E0_colors[i, j] = 0#val_return_E0(R, th, ph, icity, model_param, 1)

    img.set_data(colors)
    img.set_clim(vmin=0, vmax=np.max(colors))

    F_z = force_calc(frame, 1.5, A, B)
    F_z = int(F_z * 100) / 100

    # Draw integral curves
    if False:
        for i in range(seeds):
            r_val, theta0, phi0 = seeds_r[i], seeds_theta[i], seeds_phi[i]

            vect_c = rka_iter(r_val, theta0, phi0, num_its, icity,
                            ending_tolerance, delta_0, safety, h0, model_param)

            its = vect_c.its
            thetas = np.array(vect_c.y[0:its])
            phis   = np.array(vect_c.z[0:its])

            lines[i].set_data(phis, thetas)

    ax.set_title(r'Max $\ell$ coef = %s | $F_z$ = %s | $\Omega$ = %s$' % (int(model_param.ell), F_z, 1))
    print('Time', dt.now() - start_frame, frame, "% min:", np.min(colors), "max:", np.max(colors))

    return [img, *lines]

# ------------------ Run animation ------------------
frames = [5,10,15,20,25,30,35,40,45,50,55]
ani = FuncAnimation(fig, update, frames=frames, blit=False)

title = 'dTheta_%s_lMax_%s' % (str(gauss_dtheta).replace(".", "p"), max(frames))
title += ".gif"

print('Total time', dt.now() - start)

ani.save(title, fps=1, dpi=200)