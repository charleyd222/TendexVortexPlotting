import matplotlib.pyplot as plt
import numpy as np
from ctypes import *
from datetime import datetime as dt
from util import vect, vis_params, spherical_fibonacci_points, fit_circle
from sklearn.cluster import DBSCAN

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

# Make data

model_param = vis_params(
    M=1.,
    omega1=1,
    omega2=-1.,
    t=0.,          # this will be animated
    C2=1.5,
    w2=0
)
icity = 1
l = 500

# Resolution of the sphere grid
n_theta = 2*l
n_phi   = l
R   = 1.0

# Draw lines
num = 500
seeds_r, seeds_theta, seeds_phi = spherical_fibonacci_points(num, R)
seeds = len(seeds_r)
num_its = 5
delta_0 = 10e-6
h0 = 10e-2
safety = .9
ending_tolerance = .1

theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2*np.pi, n_phi)
Phi, Theta = np.meshgrid(phi, theta)

# Color the sphere
colors = np.zeros_like(Theta)
for i in range(Theta.shape[0]):
    for j in range(Theta.shape[1]):
        th = Theta[i, j]
        ph = Phi[i, j]
        colors[i, j] = val_return(R, th, ph, icity, model_param, 1)


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot()

# Label axes
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$\theta$')
ax.set_title('Sphere S projected down. Current octopole')
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(0, np.pi)

# Draw integral curves
for i in range(seeds):
    r_val, theta0, phi0 = seeds_r[i], seeds_theta[i], seeds_phi[i]

    vect_c = rka_iter(r_val, theta0, phi0, num_its, icity,
                      ending_tolerance, delta_0, safety, h0, model_param)

    its = vect_c.its
    thetas = np.array(vect_c.y[0:its])
    phis   = np.array(vect_c.z[0:its])

    ax.plot(phis, thetas, color='black', linewidth=0.6, alpha=0.9)


# Heatmap uses phi horizontally and theta vertically
m = ax.imshow(
    colors,
    extent=[0, 2*np.pi, 0, np.pi],
    origin='lower',
    aspect='auto',
    cmap='viridis'
)

dPhi = .01
edge = 0
dTheta = dPhi/2
phi_min, phi_max = 0+edge,2*np.pi-edge
theta_min, theta_max = 0+edge, np.pi-edge
vect_c = singular_find(R, icity, ending_tolerance, delta_0, safety, h0, model_param, dPhi, dTheta, 1, phi_min, phi_max, theta_min, theta_max)
its = vect_c.its
x = vect_c.x[0:its]
y = vect_c.y[0:its]
#plt.scatter(x,y)

points = np.column_stack([x, y])  # your array

eps = .1
db = DBSCAN(eps=eps, min_samples=7).fit(points)
labels = db.labels_

x_cent = []
y_cent = []
for label in set(labels):
    if label == -1:
        continue

    pts = points[labels == label]
    xc, yc, r = fit_circle(pts[:,0], pts[:,1])
    if r < .2:
        t = np.linspace(0,2*np.pi, 100)
        x_cent += [xc]
        y_cent += [yc]
        plt.plot(r*np.cos(t)+xc, r*np.sin(t)+yc)

plt.scatter(x_cent, y_cent)
plt.scatter(x, y)


if icity == 1:
    fig.colorbar(m, label=r'$\lambda_+$', ax=ax)
else:
    fig.colorbar(m, label=r'$\lambda_-$', ax=ax)

fig.tight_layout()


print(dt.now() - start)
plt.show()
