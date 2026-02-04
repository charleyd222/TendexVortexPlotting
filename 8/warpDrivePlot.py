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
seed_distance = 1
title, x0, y0, z0 = seed(6, seed_distance,70) # 0: Plane 1: Circular 2: Spherical 3: Helical 4: Random
#x0 = [-.5]
#y0 = [-.5]
#z0 = [-.5]
curve = {'x':[],'y':[],'z':[],'f':[],'m':[],'c':[],'lg':[]}
seeds = len(x0)
num_its = 50
delta_0 = 10e-6
h0 = 0.3
safety = .9
ending_tolerance = .01
pos_color = 'red'
neg_color = 'blue'
icity = 1
R = 1
sigma = 1
vX = 1

twist = True

# load C++
EBool = True
if EBool:
    rka_iter = CDLL("./cppScripts/EFieldCalc").rka_iter
else:
    rka_iter = CDLL("./cppScripts/BFieldCalc").rka_iter
rka_iter.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double, c_int, c_int, c_double, c_double, c_double, c_double, c_double]
rka_iter.restype = vect

# Start plot
fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')

#Color maps
norm = mcolors.SymLogNorm(.1, vmin=-10e6, vmax=10e6)
#norm = mcolors.LogNorm(10e-3, vmax=1)
#norm = mcolors.Normalize(vmin=-1, vmax=1)

cmap = mcolors.LinearSegmentedColormap.from_list('red_grey_blue', ['red', 'grey', 'blue'])
hAvgTotal = 0
for i in range(seeds):
    for icity in [1]:
        for factor in [1,-1]:
            # Starting point of each field line
            x, y, z = x0[i], y0[i], z0[i]
        
            vect_c = rka_iter(R, sigma, vX, x, y, z, num_its, icity, ending_tolerance, delta_0, safety, h0, factor)
            its = vect_c.its
            x = vect_c.x[0:its]
            y = vect_c.y[0:its]
            z = vect_c.z[0:its]
            hAvgTotal += vect_c.hAvg
            if twist:
                m = np.array(vect_c.twist[0:its])
            else:
                m = np.abs(np.array(vect_c.m[0:its])) * icity

            lc = colorline(ax, x, y, z, m, width=1.5, norm = norm, cmap=cmap)

print(dt.now() - start) 
print('Average h:',hAvgTotal/i)
lim = R*1.5

ax.set_xlim(-lim,lim)
ax.set_ylim(-lim,lim)
ax.set_zlim(-lim,lim)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

EorB = ' B '
if EBool:
    EorB = ' E '
    

ax.set_title('Warp Drive'+EorB+'Field. Sigma: %s  R: %s  Seed Distance: %s' %(sigma, R, seed_distance))
if twist:
    label = 'twist'
else:
    label = 'eigenvalue'
fig.colorbar(cm.ScalarMappable(cmap=cmap, norm = norm), ax=ax, label = label)
#pos_ax.scatter(x0, y0, z0)

plt.show()
fig.savefig('neg.png', dpi=300)