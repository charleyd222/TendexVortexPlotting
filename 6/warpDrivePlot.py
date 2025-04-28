from ctypes import *
from datetime import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

from util.plot import colorline
from util.seeding import seed


#Runtime
start = dt.now()

class vect(Structure):
    _fields_ = [('x', c_double*10000), ('y', c_double*10000), ('z', c_double*10000), ('m', c_double*10000), ('its', c_int)]

#Value setup
title, x0, y0, z0 = seed(6, 1, 100) # 0: Plane 1: Circular 2: Spherical 3: Helical 4: Random
#x0 = [-.5]
#y0 = [-.5]
#z0 = [-.5]
curve = {'x':[],'y':[],'z':[],'f':[],'m':[],'c':[],'lg':[]}
seeds = len(x0)
num_its = 25
delta_0 = 10e-6
h0 = 10e-3
safety = .9
ending_tolerance = .01
pos_color = 'red'
neg_color = 'blue'
icity = 1
R = 1
sigma = 1
vX = 1

# load C++
EBool = False
if EBool:
    rka_iter = CDLL("./cppScripts/EFieldCalc").rka_iter
else:
    rka_iter = CDLL("./cppScripts/BFieldSymCalc").rka_iter
rka_iter.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double, c_int, c_int, c_double, c_double, c_double, c_double, c_double]
rka_iter.restype = vect

# Start plot
fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')

#Color maps
norm = mcolors.SymLogNorm(.01, vmin=-10e1, vmax=10e1)
cmap = mcolors.LinearSegmentedColormap.from_list('red_grey_blue', ['red', 'grey', 'blue'])

for i in range(seeds):
    for icity in [-1,1]:
        for factor in [1]:
            # Starting point of each field line
            x, y, z = x0[i], y0[i], z0[i]
        
            vect_c = rka_iter(R, sigma, vX, x, y, z, num_its, icity, ending_tolerance, delta_0, safety, h0, factor)
            its = vect_c.its
            x = vect_c.x[0:its]
            y = vect_c.y[0:its]
            z = vect_c.z[0:its]
            m = np.abs(np.array(vect_c.m[0:its])) * icity

            lc = colorline(ax, x, y, z, m, norm = norm, cmap=cmap)

print(dt.now() - start) 

lim = R*1.5

ax.set_xlim(-lim,lim)
ax.set_ylim(-lim,lim)
ax.set_zlim(-lim,lim)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
if EBool:
    ETitle = 'E'
else:
    ETitle = 'B'
ax.set_title('Warp Drive ' + ETitle + ' Field')
fig.colorbar(cm.ScalarMappable(cmap=cmap, norm = norm), ax=ax, label = 'eigenvalue')
#pos_ax.scatter(x0, y0, z0)

plt.show()
fig.savefig('neg.png', dpi=300)