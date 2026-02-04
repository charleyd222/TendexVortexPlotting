from ctypes import *
from datetime import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import matplotlib.animation as animation
from functools import partial

from util.plot import colorline
from util.seeding import seed

#Runtime
start = dt.now()

class vect(Structure):
    _fields_ = [('x', c_double*10000), ('y', c_double*10000), ('z', c_double*10000), ('m', c_double*10000), ('its', c_int)]

# load C++
rka_iter = CDLL("./cppScripts/rka_iter_custT").rka_iter_double
rka_iter.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double, c_int, c_int, c_double, c_double, c_double, c_double]
rka_iter.restype = vect

# Value setup
num_its = 30 # Iterations per seed point
delta_0 = 10e-5 # Min deviation from next step
h0 = 10e-4 #Starting step size
safety = .98 # Scaling factor for step size change
ending_tolerance = 1.0 # How far from poles to step simulation
icity = 1

# Color maps
cmap = mcolors.LinearSegmentedColormap.from_list('pos', [('lightsteelblue'), ('navy')], N=100)
norm = mcolors.LogNorm(vmin=10e-9, vmax=10e1)

# Animation setup
lim = 25 # x y bounds for plot
frame_num = 100 # Num frames for animation
seps = np.linspace(10,1,frame_num//2)
seps = np.append(seps, np.logspace(0,-4,frame_num//2))
t_list = np.linspace(.4,.6,frame_num)
artists = []

def frame(frame, t=None, sep=None):
    title, x0, y0, z0 = seed(4) # 0: Plane 1: Circular 2: Spherical 3: Helical 4: Random
    seeds = len(x0)

    ax.cla()
    if sep == None:
        sep = seps[frame]
    if t == None:
        t = t_list[frame]
    for i in range(seeds):
        # Starting point of each field line
        x, y, z = x0[i], y0[i], z0[i]
    
        vect_c = rka_iter(t*np.pi, sep, sep, x, y, z, num_its, icity, ending_tolerance, delta_0, safety, h0)
        its = vect_c.its
        x = vect_c.x[0:its]
        y = vect_c.y[0:its]
        z = vect_c.z[0:its]
        m = np.abs(np.array(vect_c.m[0:its]))

        lc = colorline(ax, x, y, m, cmap='jet', norm = norm)

    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)

    title = 'pi* ' + str(int(((t))*100)/100) +' S:' + str(int(sep))
    fig.suptitle(title)
    artists.append([ax])

    return [ax]
    

# Start plot
fig, ax = plt.subplots(1)
fig.colorbar(cm.ScalarMappable(cmap='jet', norm = norm), ax=ax, label = 'pos eigenvalue')

# Setup animation
if False:
    title = 'IterAngleSmall'
    ani = animation.FuncAnimation(fig, partial(frame, sep = 0.001), frames=frame_num)
else:
    title = 'IterSeps'
    ani = animation.FuncAnimation(fig, partial(frame, t = .5), frames=frame_num)
ani.save('plots/ani'+title+'.gif', writer='pillow', fps=10)


print(title + ': ' + str(dt.now() - start))