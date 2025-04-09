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
title, x0, y0, z0 = seed(4,1) # 0: Plane 1: Circular 2: Spherical 3: Helical 4: Random
#x0 = [-.5]
#y0 = [-.5]
#z0 = [-.5]
curve = {'x':[],'y':[],'z':[],'f':[],'m':[],'c':[],'lg':[]}
seeds = len(x0)
num_its = 500
delta_0 = 10e-5
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
if False:
    rka_iter = CDLL("./cppScripts/EFieldCalc").rka_iter
else:
    rka_iter = CDLL("./cppScripts/BFieldCalc").rka_iter
rka_iter.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double, c_int, c_int, c_double, c_double, c_double, c_double]
rka_iter.restype = vect

# Start plot
neg_fig = plt.figure(1)
neg_ax = neg_fig.add_subplot(projection='3d')
pos_fig = plt.figure(1)
pos_ax = pos_fig.add_subplot(projection='3d')

#Color maps
pos_cmap = mcolors.LinearSegmentedColormap.from_list('pos', [('lightsteelblue'), ('navy')], N=100)
neg_cmap = mcolors.LinearSegmentedColormap.from_list('pos', [('mistyrose'), ('darkred')], N=100)
blank_pos_cmap = mcolors.LinearSegmentedColormap.from_list('pos', [('blue'), ('blue')], N=1)
blank_neg_cmap = mcolors.LinearSegmentedColormap.from_list('pos', [('red'), ('red')], N=1)
pos_norm = mcolors.LogNorm(vmin=10e-5, vmax=10e0)
neg_norm = mcolors.LogNorm(vmin=10e-5, vmax=10e0)
width = False
color = True

for i in range(seeds):
    #print(i)
    for icity in [1]:
        # Starting point of each field line
        x, y, z = x0[i], y0[i], z0[i]
    
        vect_c = rka_iter(R, sigma, vX, x, y, z, num_its, icity, ending_tolerance, delta_0, safety, h0)
        its = vect_c.its
        x = vect_c.x[0:its]
        y = vect_c.y[0:its]
        z = vect_c.z[0:its]
        m = np.abs(np.array(vect_c.m[0:its]))
        #print(min(m))
        #print(max(m))
        #print()

        if icity == 1:
            if width:
                lc = colorline(pos_ax, x, y, z, m, norm = pos_norm, widths = m, cmap=blank_pos_cmap)
            elif color:
                lc = colorline(pos_ax, x, y, z, m, norm = pos_norm, cmap=pos_cmap)
        else:
            if width:
                lc = colorline(pos_ax, x, y, z, m, norm = neg_norm, widths = m, cmap=blank_neg_cmap)
            elif color:
                lc = colorline(pos_ax, x, y, z, m, norm = neg_norm, cmap=neg_cmap)

print(dt.now() - start) 

#if color:
#    pos_fig.colorbar(cm.ScalarMappable(cmap=pos_cmap, norm = pos_norm), ax=pos_ax)
#    neg_fig.colorbar(cm.ScalarMappable(cmap=neg_cmap, norm = neg_norm), ax=neg_ax)

lim = 1.5

pos_ax.set_xlim(-lim,lim)
pos_ax.set_ylim(-lim,lim)
pos_ax.set_zlim(-lim,lim)

neg_ax.set_xlim(-lim,lim)
neg_ax.set_ylim(-lim,lim)

pos_fig.savefig('pos.png', dpi=300)

plt.show()
#neg_fig.savefig('neg.png', dpi=300)