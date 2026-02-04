import matplotlib.pyplot as plt
import numpy as np
from ctypes import *
from datetime import datetime as dt

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

def make_seeds(N, R, rand=False, cart = False):
    if rand:
        theta = np.random.rand(N) * 2 * np.pi
        u = np.random.rand(N)
    else:
        theta = np.linspace(0,2 * np.pi, N)
        u =  np.linspace(0,1,N)

    phi = np.acos(1 - (2*u))
    r = np.zeros(N) + R
    #theta = np.full(N,.5)

    if cart:
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)

        return x, y, z

    return r, theta, phi

#Runtime
start = dt.now()

class vect(Structure):
    _fields_ = [('x', c_double*10000),
                ('y', c_double*10000),
                ('z', c_double*10000), 
                ('m', c_double*10000),
                ('its', c_int)]
    
class vals(Structure):
    _fields_ = [('M', c_double),
                ('omega', c_double),
                ('t', c_double), 
                ('C', c_double),
                ('w', c_double)]

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

# Load cpp
rka_iter = CDLL("./cppScripts/rka_iter").rka_iter
val_return = CDLL("./cppScripts/rka_iter").val_return

rka_iter.argtypes = [c_double, c_double, c_double, c_int, c_int, c_double, c_double, c_double, c_double, vals]
val_return.argtypes = [c_double, c_double, c_double, c_int, vals, c_int]
rka_iter.restype = vect
val_return.restype = c_double

# Make data
model_param = vals(
    M = 1.,
    omega = 2.,
    t = 0.,
    C = .75,
    w = - np.pi/4
)
icity = 1
l = 50

# Resolution of the sphere grid
n_theta = 2*l
n_phi   = l
R   = 1.0

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

# Draw lines
num = 300
seeds_r, seeds_theta, seeds_phi = make_seeds(num, R, cart=False, rand = True)
seeds_r, seeds_theta, seeds_phi = spherical_fibonacci_points(num, R)
seeds = len(seeds_r)
num_its = 5
delta_0 = 10e-2
h0 = 10e-2
safety = .9
ending_tolerance = 1e-5
pos_color = 'red'
neg_color = 'blue'
icity = -1

for i in range(seeds):
    r_val, theta, phi = seeds_r[i], seeds_theta[i], seeds_phi[i]

    vect_c = rka_iter(r_val, theta, phi, num_its, icity, ending_tolerance, delta_0, safety, h0, model_param)
    its = vect_c.its
    r_vals = np.array(vect_c.x[0:its])
    thetas = np.array(vect_c.y[0:its])
    phis = np.array(vect_c.z[0:its])
    m = np.array(vect_c.m[0:its])

    x = r_vals * np.sin(thetas) * np.cos(phis)
    y = r_vals * np.sin(thetas) * np.sin(phis)
    z = r_vals * np.cos(thetas)

    ax.plot(x,y,z, c='black', ms=1, zorder = 3, alpha=1)
    
#ax.scatter(x,y,z, zorder=2)
X = R * np.sin(Theta) * np.cos(Phi)
Y = R * np.sin(Theta) * np.sin(Phi)
Z = R * np.cos(Theta)

# Normalize values to [0,1]
normed = colors
vmin, vmax = colors.min(), colors.max()
normed = (colors - vmin) / (vmax - vmin + 1e-12)

# Convert normalized values → RGBA  (this returns an (M,N,4) array)
colors_temp = plt.cm.viridis(normed)
#norm = mcolors.Normalize(vmin=0, vmax=np.max(colors))
#cmap = mat.colormaps['jet']
#colors_temp = cmap(norm(normed))

# Plot sphere
surf = ax.plot_surface(
    X, Y, Z,
    facecolors=colors_temp,
    rstride=1, cstride=1,
    linewidth=0,
    antialiased=False,
    shade=False
)

# Colorbar
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_clim(vmin, vmax)
plt.colorbar(mappable, shrink=0.6, label='value', ax=ax)

ax.set_box_aspect([1,1,1])
plt.show()