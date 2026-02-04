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
    _fields_ = [('x', c_double*10000), 
                ('y', c_double*10000), 
                ('z', c_double*10000), 
                ('m', c_double*10000), 
                ('twist', c_double*10000), 
                ('its', c_int), 
                ('hAvg', c_double)]

#Value setup
dist = 1
title, x0, y0, z0 = seed(4,dist,50) # 0: Plane 1: Circular 2: Spherical 3: Helical 4: Random
curve = {'x':[],'y':[],'z':[],'f':[],'m':[],'c':[],'lg':[]}
seeds = len(x0)
num_its = 100
delta_0 = 10e-2
h0 = .002
safety = .98
ending_tolerance = .2
w = .5

# load C++
rka_iter = CDLL("./cppScripts/timeDependentE").rka_iter
rka_iter.argtypes = [c_double, c_double, c_double, c_int, c_int, c_double, c_double, c_double, c_double, c_double]
rka_iter.restype = vect

# Start plot
fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')

#Color maps
norm = mcolors.SymLogNorm(.01, vmin=-10e1, vmax=10e1)
cmap = mcolors.LinearSegmentedColormap.from_list('red_grey_blue', ['red', 'grey', 'blue'])

hAvgTotal = 0
for i in range(seeds):
    #print(i)
    for icity in [1]:
        for factor in [1,-1]:
            # Starting point of each field line
            x, y, z = x0[i], y0[i], z0[i]
        
            vect_c = rka_iter(x, y, z, num_its, icity, ending_tolerance, delta_0, safety, h0, w)

            its = vect_c.its
            x = vect_c.x[0:its]
            y = vect_c.y[0:its]
            z = vect_c.z[0:its]
            m = np.array(vect_c.twist[0:its])
            hAvgTotal += vect_c.hAvg

            lc = colorline(ax, x, y, z, m, norm = norm, cmap=cmap)

print(dt.now() - start) 
print('Average h:',hAvgTotal/i)


lim = dist*1.5

ax.set_xlim(-lim,lim)
ax.set_ylim(-lim,lim)
ax.set_zlim(-lim,lim)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Rotating E Field, w: %s' %(w))
fig.colorbar(cm.ScalarMappable(cmap=cmap, norm = norm), ax=ax, label = 'twist')

plt.show()
fig.savefig('neg.png', dpi=300)

#neg_fig.savefig('neg.png', dpi=300)