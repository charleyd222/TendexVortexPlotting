#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''


'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime as dt
from util.plot import colorline
from util.seeding import seed
from ctypes import c_double, c_int, Structure, CDLL

#Runtime
start = dt.now()

# RKA setup
seed_its = 50 # Iterations per seed point
delta_0 = 10e-6 # Min deviation from next step
h0 = 10e-4 #Starting step size
safety = .98 # Scaling factor for step size change
ending_tolerance = .001 # How far from poles to step simulation
icity = -1

bound_seed_its = 1
bound_seeds = 1000
bound_seed_sizeX = 6
bound_seed_sizeY = 6
bound_seed_center_X = 0
bound_seed_center_Y = 0


# Physical setup
sep = 3
theta = 0

# Plotting Setup
distance = 5
seed_num = 1000
fig = plt.figure()
ax = [fig.add_subplot(211), fig.add_subplot(212)]
norm = mcolors.LogNorm(vmin=10e-5, vmax=10e5)


class vect(Structure):
    _fields_ = [('x', c_double*10000), 
                ('y', c_double*10000),
                ('z', c_double*10000), 
                ('m', c_double*10000),
                ('hArray', c_double*10000), 
                ('det', c_double*10000), 
                ('its', c_int)]
    

rka_iter = CDLL("./cppScripts/rka_iter_CustT_2Quad").rka_iter_double
#rka_iter = CDLL("./cppScripts/rka_iter_classical").rka_iter_double

rka_iter.argtypes = [c_double, c_double, c_double, c_double, 
                     c_double, c_double, c_int, c_int, c_double, 
                     c_double, c_double, c_double]
rka_iter.restype = vect

title, x_seeds_bound, y_seeds_bound, z_seeds_bound = seed(6, (bound_seed_sizeX, bound_seed_sizeY), 
                                                  bound_seeds, (bound_seed_center_Y, bound_seed_center_X))

show_full = True

title, x_seeds, y_seeds, z_seeds = seed(4, distance, seed_num)
seeds = len(x_seeds)

if show_full:
    for i in range(seeds):
        x, y, z = x_seeds[i], y_seeds[i], z_seeds[i]
        
        vect_c = rka_iter(theta*np.pi, sep, sep, x, y, z, 
                          seed_its, icity, ending_tolerance, delta_0, safety, h0)
        
        its = vect_c.its
        x = vect_c.x[0:its]
        y = vect_c.y[0:its]
        z = vect_c.z[0:its]
        
        colorline(ax[0], x, y, [2], norm=norm, width=1.5)
    
first = True
v_first = 0
dots = []
dotsX = []
for i in range(len(x_seeds_bound)):
    x, y, z = x_seeds_bound[i], y_seeds_bound[i], z_seeds_bound[i]
    
    vect_c = rka_iter(theta*np.pi, sep, sep, x, y, z, 
                      bound_seed_its, icity, ending_tolerance, delta_0, safety, h0)
    
    its = vect_c.its
    x = vect_c.x[0:its]
    y = vect_c.y[0:its]
    z = vect_c.z[0:its]
    
    if its > bound_seed_its -1:
        t = np.arctan2(y,x)

        dots += [np.mean(t)]
        dotsX += [i]
    
    #colorline(ax[0], x, y, [2], norm=norm, width=1.5)
        
    
#dots = np.unwrap(dots)
#ax[1].plot(dots)
dotsX = np.array(dotsX)
ax[1].scatter(np.array(100*dotsX/len(dotsX)), np.array(dots)/np.pi, s=1, zorder=1)
ax[0].scatter(x_seeds_bound, y_seeds_bound, s=1.5)
ax[0].set_xlim(-(4/3)*distance, (4/3)*distance)
ax[0].set_ylim(-(4/3)*distance, (4/3)*distance)

ax[0].set_title('Boundary in Blue')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')

ax[1].set_title('Radians rotated from vertical')
ax[1].set_ylabel('Radians')
ax[1].set_xlabel('% around boundary ')
fig.tight_layout()

print(dt.now() - start)

plt.show()