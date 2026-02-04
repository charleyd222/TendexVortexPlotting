from ctypes import *
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from util.seeding import seed
from util.plot import colorline
from determinant.determinant import determinant
import matplotlib.colors as mcolors

detCalc = determinant()

#Runtime
start = dt.now()

class vect(Structure):
    _fields_ = [('x', c_double*10000), 
                ('y', c_double*10000),
                ('z', c_double*10000), 
                ('m', c_double*10000),
                ('hArray', c_double*10000), 
                ('det', c_double*10000), 
                ('its', c_int)]




def DoubleQuadroPlot():
    pass

# load C++
rka_iter = CDLL("./cppScripts/rka_iter_classical").rka_iter_double
rka_iter.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double, c_int, c_int, c_double, c_double, c_double, c_double]
rka_iter.restype = vect

# RKA setup
num_its = 30 # Iterations per seed point
delta_0 = 10e-6 # Min deviation from next step
h0 = 10e-4 #Starting step size
safety = .98 # Scaling factor for step size change
ending_tolerance = .001 # How far from poles to step simulation
icity = -1

# Physical setup
sep = 0
sepY = 3
theta = 0

# Plotting Setup
distance = 6
title, x_seeds, y_seeds, z_seeds = seed(4, distance, 1000) # 0: Plane 1: Circular 2: Spherical 3: Helical 4: Random
seeds = len(x_seeds)
fig, ax = plt.subplots(1)
# Color maps
cmap = mcolors.LinearSegmentedColormap.from_list('pos', [('lightsteelblue'), ('navy')], N=100)
norm = mcolors.LogNorm(vmin=10e-5, vmax=10e5)
norm2 = mcolors.Normalize(vmin=0, vmax=1)

for i in range(seeds):
    # Starting point of each field line
    x, y, z = x_seeds[i], y_seeds[i], z_seeds[i]

    vect_c = rka_iter(theta*np.pi, sep, sepY, x, y, z, num_its, icity, ending_tolerance, delta_0, safety, h0)
    its = vect_c.its
    x = vect_c.x[0:its]
    y = vect_c.y[0:its]
    z = vect_c.z[0:its]
    h = vect_c.hArray[0:its]
    #det = vect_c.det[0:its]
    m = np.abs(np.array(vect_c.m[0:its]))

    if len(x) > 2 and len(y) > 2:
        x0 = np.array(x[0:-2])
        x1 = np.array(x[1:-1])
        x2 = np.array(x[2:])
        y0 = np.array(y[0:-2])
        y1 = np.array(y[1:-1])
        y2 = np.array(y[2:])
        
        Dx = .5 * (x2 - x0)
        D2x = (x0 - 2*x1 + x2)
        Dy = .5 * (y2 - y0)
        D2y = (y0 - 2*y1 + y2)

        # Numerator: (x'y'' - y'x'')
        num = np.abs((Dx * D2y) - (Dy * D2x))
        # Denominator: (x'^2 + y'^2)^(3/2)
        den = ((Dx)**2 + (Dy)**2)**1.5
                    
        curvature = num / den
        m = m[1:-1]

        colorline(ax, x, y, [1], norm = norm2, cmap='jet', width=1.5)
        

    #Z = detCalc.det(lim, sep, t)
    #ax.imshow(Z, extent=(*[-3, 3], *[-3, 3]), origin='lower',
    #    aspect='equal', cmap = 'jet', norm=detCalc.detNormAbs)
        #colorline(ax, x[1:-1], y[1:-1], group, norm = norm2, cmap='jet', width=1.5)
    


#ax.scatter(x0, y0)
ax.set_xlim(-(4/3)*distance, (4/3)*distance)
ax.set_ylim(-(4/3)*distance, (4/3)*distance)
plt.show()

    