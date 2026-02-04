import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ctypes import *
from datetime import datetime as dt
from util import vect, rk_params, vis_params
from sklearn.cluster import DBSCAN

def spherical_to_xyz(r, theta, phi):
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)
    return x, y, z

def spherical_fibonacci_points(N, R):
    """
    Generate N points evenly distributed on a unit sphere.
    Returns theta (polar angle) and phi (azimuthal angle) arrays.
    """
    # Golden ratio
    gr = (1 + np.sqrt(5)) / 2

    # Index array
    i = np.arange(N)

    # z goes from 1 to -1
    z = 1 - (2*i + 1)/N
    theta = np.arccos(z)  # polar angle

    phi = 2 * np.pi * i / gr  # azimuthal angle
    phi = np.mod(phi, 2*np.pi)  # ensure 0 <= phi < 2pi

    r = np.zeros(N) + R

    return r, theta, phi

def make_seeds(N, R, phi_min = 0, theta_min = 0, phi_max = 2*np.pi, theta_max = np.pi):
    N = int(np.sqrt(N))
    theta = np.linspace(theta_min, theta_max, N)
    phi = np.linspace(phi_min, phi_max, N)

    theta, phi = np.meshgrid(theta, phi)

    theta = theta.flatten()
    phi = phi.flatten()

    r = np.zeros(len(theta)) + R
    return r, theta, phi

def fit_circle(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x**2 + y**2

    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = c[0], c[1]
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r


#Runtime
start = dt.now()

# Load cpp
rka_iter = CDLL("./cppScripts/rka_iter").rka_iter
val_return = CDLL("./cppScripts/rka_iter").val_return
singular_find = CDLL("./cppScripts/rka_iter").singular_find

rka_iter.argtypes = [c_double, c_double, c_double, c_int, c_int, c_double, c_double, c_double, c_double, vis_params]
val_return.argtypes = [c_double, c_double, c_double, c_int, vis_params, c_int]
singular_find.argtypes = [c_double, c_int, c_double, c_double, c_double, c_double, vis_params, c_double, c_double, c_int, c_double, c_double, c_double, c_double]

rka_iter.restype = vect
val_return.restype = c_double
singular_find.restype = vect


# ------------------ Parameters ------------------
model_param = vis_params(
    M=1.,
    omega1=1.,
    omega2=.9,
    t=0., # this will be animated
    C2=1,
    w2=-np.pi/8
)

centroid_history = {
    "t": [],
    "x": [],
    "y": [],
    "param": model_param
}

# Animation Parameters
t_start = 25
t_end = 40
frame_num = 50
fps = frame_num/5
cmax = 1.7

# Simulation Parameters
icity = 1
R = 1.0
num_seeds = 5000
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
edge = .01 # Edge where to find no singularity

# Grid Setup
phi_min, phi_max = 0+edge,2*np.pi-edge
theta_min, theta_max = 2.1+edge, np.pi-edge
theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2*np.pi, n_phi)
Phi, Theta = np.meshgrid(phi, theta)

# Sim seeds
seeds_r, seeds_theta, seeds_phi = make_seeds(num_seeds, R, phi_min = phi_min, theta_min=theta_min, phi_max = phi_max, theta_max=theta_max)
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
fig.suptitle('1: Mass Quadropole 2: Current Octopole')
cbar.set_label(r'$\lambda_+$' if icity == 1 else r'$\lambda_-$')

# ------------------ Animation update ------------------
def update(frame):
    start = dt.now()
    # Animate parameter
    model_param.t = frame

    # Color the sphere
    for i in range(Theta.shape[0]):
        for j in range(Theta.shape[1]):
            th = Theta[i, j]
            ph = Phi[i, j]
            colors[i, j] = val_return(R, th, ph, icity, model_param, 1)
    img.set_data(colors)

    # Update integral curves
    for i in range(seeds):
        r_val, theta0, phi0 = seeds_r[i], seeds_theta[i], seeds_phi[i]

        vect_c = rka_iter(
            r_val, theta0, phi0,
            num_its, icity,
            ending_tolerance, delta_0,
            safety, h0, model_param
        )

        its = vect_c.its
        thetas = np.array(vect_c.y[:its])
        phis = np.array(vect_c.z[:its])

        lines[i].set_data(phis, thetas)

    ax.set_title(r'$\Omega_1 =$ %s. $\Omega_2 =$ %s. $C_2 =$ %s. $\omega_2 =$ %s$\pi$ | t value = %s' % (model_param.omega1, model_param.omega2, model_param.C2, int((model_param.w2 / np.pi ) * 100) / 100 , int(model_param.t)))

    vect_c = singular_find(R, icity, ending_tolerance, delta_0, safety, h0, model_param, dPhi, dTheta, 1, phi_min, phi_max, theta_min, theta_max)
    its = vect_c.its
    x = vect_c.x[0:its]
    y = vect_c.y[0:its]
    points = np.c_[x, y]  # your array

    eps = .03
    if its > 0:
        db = DBSCAN(eps=eps, min_samples=5).fit(points)
        labels = db.labels_

        x_cent = []
        y_cent = []
        for label in set(labels):
            if label == -1:
                continue

            pts = points[labels == label]
            xc, yc, r = fit_circle(pts[:,0], pts[:,1])
            if r < .2:
                x_cent += [xc]
                y_cent += [yc]
        if len(x_cent) > 0:
            scat.set_offsets(np.c_[x_cent, y_cent])

            centroid_history["t"].extend([frame] * len(x_cent))
            centroid_history["x"].extend(x_cent)
            centroid_history["y"].extend(y_cent)
            

        else:
            scat.set_offsets(np.c_[[], []])
    else:
        scat.set_offsets(np.c_[[], []])

    print('Time', dt.now() - start, str(int(frame * 1000 / t_end)/10), "%. min:", np.min(colors), "max:", np.max(colors))

    return [img, *lines]

# ------------------ Run animation ------------------
print(model_param.omega2)
frames = np.linspace(t_start, t_end, frame_num)
ani = FuncAnimation(fig, update, frames=frames, blit=False)

title = 'O1_%s_O2_%s_C2_%s_w_%spi' % (model_param.omega1, model_param.omega2, model_param.C2, int((model_param.w2 / np.pi ) * 100) / 100)

title = title.replace(".","p").replace("-","n")
title += ".gif"

ani.save(title, fps=fps, dpi=200)

import pickle
with open('dataSim', 'wb') as f:
    pickle.dump(centroid_history, f) # Use the highest protocol for best performance