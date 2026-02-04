import matplotlib.pyplot as plt
import numpy as np
from ctypes import *
from datetime import datetime as dt
from util import vect, vis_params, spherical_fibonacci_points, fit_circle
from sklearn.cluster import DBSCAN

#Runtime
start = dt.now()

l = [2,3,4,5,6,7,8,9,10,11,12]
m = [2,2,2,2,2,2,2,2,2,2,2]
A = [0.290397, 0.264786, 0.203361, 0.123797, 0.043046, -0.0263431, -0.0780927, -0.111639, -0.130006, -0.137491, -0.13806]
B = [0.293507, 0.280348, 0.245926, 0.208078, 0.178072, 0.159602, 0.150825, 0.147607, 0.146125, 0.144037, 0.140469]

# Load cpp
rka_iter = CDLL("./cppScripts/rka_iter").rka_iter
val_return = CDLL("./cppScripts/rka_iter").val_return

rka_iter.argtypes = [c_double, c_double, c_double, c_int, c_int, c_double, c_double, c_double, c_double, vis_params]
val_return.argtypes = [c_double, c_double, c_double, c_int, vis_params, c_int]

rka_iter.restype = vect
val_return.restype = c_double

# Make data
model_param = vis_params(
    M=1.,
    omega1=1,
    omega2=1,
    t=0.,
    C2=.1,
    w2=np.pi/4,
    ell=100
)
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
ending_tolerance = .0001

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
ax.set_title('I between 2 and 30 linear combination, dTheta = .1')
#ax.set_xlim(0, 2 * np.pi)
#ax.set_ylim(0, np.pi)
if True:
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


print(dt.now() - start)
plt.show()
