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
                ('hArray', c_double*10000), 
                ('det', c_double*10000), 
                ('m', c_double*10000), 
                ('its', c_int)]

#Value setup
title, x0, y0, z0 = seed(4,40) # 0: Plane 1: Circular 2: Spherical 3: Helical 4: Random
curve = {'x':[],'y':[],'z':[],'f':[],'m':[],'c':[],'lg':[]}
seeds = len(x0)
num_its = 1000
delta_0 = 10e-2
h0 = 10e-2
safety = .9
ending_tolerance = 1.0
pos_color = 'red'
neg_color = 'blue'
icity = 1

# load C++
rka_iter = CDLL("./cppScripts/rka_iter").rka_iter
rka_iter.argtypes = [c_double, c_double, c_double, c_int, c_int, c_double, c_double, c_double, c_double]
rka_iter.restype = vect

# Start plot
pos_fig, pos_ax = plt.subplots(1)
neg_fig, neg_ax = plt.subplots(1)

#Color maps
pos_cmap = mcolors.LinearSegmentedColormap.from_list('pos', [('lightsteelblue'), ('navy')], N=100)
neg_cmap = mcolors.LinearSegmentedColormap.from_list('pos', [('mistyrose'), ('darkred')], N=100)
blank_pos_cmap = mcolors.LinearSegmentedColormap.from_list('pos', [('blue'), ('blue')], N=1)
blank_neg_cmap = mcolors.LinearSegmentedColormap.from_list('pos', [('red'), ('red')], N=1)
#pos_norm = mcolors.Normalize(vmin=0, vmax=1)
pos_norm = mcolors.LogNorm(vmin=10e-5, vmax=10e0)
neg_norm = mcolors.Normalize(vmin=0, vmax=1000)
width = False
color = True

aT = []
for i in range(seeds):
    for icity in [1]:
        # Starting point of each field line
        x, y, z = x0[i], y0[i], z0[i]
    
        vect_c = rka_iter(x, y, z, num_its, icity, ending_tolerance, delta_0, safety, h0)
        its = vect_c.its
        x = vect_c.x[0:its]
        y = vect_c.y[0:its]
        z = vect_c.z[0:its]
        h = vect_c.hArray[0:its]
        det = vect_c.det[0:its]
        m = np.abs(np.array(vect_c.m[0:its]))

        if len(x) > 3 and len(y) > 3:
            r = np.array([x,y]).T
            N = r.shape[0]
            curvature = np.full(N, 0.)

            # First derivative (central difference)
            rp = np.gradient(r, h, axis=0)
            # Second derivative (central difference)
            rpp = rp = np.gradient(rp, h, axis=0)

            # Numerator: determinant (x'y'' - y'x'')
            num = np.abs(rp[:, 0] * rpp[:, 1] - rp[:, 1] * rpp[:, 0])
            # Denominator: (x'^2 + y'^2)^(3/2)
            den = (rp[:,0]**2 + rp[:,1]**2)**1.5
                        
            curvature = num / den

        else:
            curvature = np.zeros(len(x))


        if icity == 1:
            if width:
                lc = colorline(pos_ax, x, y, m, norm = pos_norm, widths = m, cmap=blank_pos_cmap)
            elif color:
                lc = colorline(pos_ax, x, y, curvature, norm = pos_norm, cmap='jet')
        else:
            if width:
                lc = colorline(neg_ax, x, y, m, norm = pos_norm, widths = m, cmap=blank_neg_cmap)
            elif color:
                lc = colorline(neg_ax, x, y, m, norm = pos_norm, cmap='jet')

        if False:
            fig, ax = plt.subplots(ncols=3,nrows=1)
            ax[0].plot(a[0],a[1])
            ax[1].set_title('curvature')
            ax[1].plot(curvature[0],curvature[1])
            ax[2].set_title('y')
            ax[2].plot(aY[0],aY[1])
            plt.show()
            1/0

print(dt.now() - start)

if color:
    pos_fig.colorbar(cm.ScalarMappable(cmap='jet', norm = pos_norm), ax=pos_ax, label='curvature')
    neg_fig.colorbar(cm.ScalarMappable(cmap='jet', norm = neg_norm), ax=neg_ax, label='curvature')

pos_ax.set_xlim(-120,120)
pos_ax.set_ylim(-120,120)
pos_ax.set_title('Quadropole. Positive EVals.')
neg_ax.set_xlim(-120,120)
neg_ax.set_ylim(-120,120)
neg_ax.set_title('Quadropole. Negative EVals.')


pos_fig.savefig('plots/quadropole/quadPos1.png', dpi=500)
neg_fig.savefig('plots/quadropole/quadNeg.png', dpi=500)