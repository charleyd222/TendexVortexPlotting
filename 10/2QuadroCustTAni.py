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
from determinant.determinant import determinant
detCalc = determinant()

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

# load C++
rka_iter = CDLL("./cppScripts/rka_iter_custT_2Quad").rka_iter_double
rka_iter.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double, c_int, c_int, c_double, c_double, c_double, c_double]
rka_iter.restype = vect


# Value setup
num_its = 10 # Iterations per seed point
delta_0 = 10e-6 # Min deviation from next step
h0 = 10e-3 #Starting step size
safety = .98 # Scaling factor for step size change
ending_tolerance = .00001 # How far from poles to step simulation
icity = 1

# Color maps
cmap = mcolors.LinearSegmentedColormap.from_list('pos', [('lightsteelblue'), ('navy')], N=100)
norm = mcolors.LogNorm(vmin=10e-5, vmax=10e5)



# Animation setup
lim = 4 # x y bounds for plot
offset_x = 0
offset_y = 0
frame_num = 1 # Num frames for animation
#seps = np.linspace(10,1,frame_num//2)
#seps = np.append(seps, np.logspace(0,-4,frame_num//2 + frame_num % 2))
seps = np.linspace(.1,3,frame_num)
t_list = np.linspace(.4,.6,frame_num)
artists = []

#global curvature_total
#global eigen_total
curvature_total = {}
eigen_total = {}
dCdt = {}
totals = {}

for n in range(frame_num):
    totals[n] = (curvature_total.copy(), dCdt.copy(), eigen_total.copy())

def frame(frame, totals=None, t=None, sep=None, detCalc=None, lim=None):
    ax.cla()
    
    if sep == None:
        sep = seps[frame]
    if t == None:
        t = t_list[frame]
    fac = 4/5
    title, x0, y0, z0 = seed(4, -fac*lim, fac*lim, -fac*lim, fac*lim, 1000) # 0: Plane 1: Circular 2: Spherical 3: Helical 4: Random
    title=''
    seeds = len(x0)

    for i in range(seeds):
        # Starting point of each field line
        x, y, z = x0[i], y0[i], z0[i]
    
        vect_c = rka_iter(t*np.pi, sep, sep, x, y, z, num_its, icity, ending_tolerance, delta_0, safety, h0)
        its = vect_c.its
        x = vect_c.x[0:its]
        y = vect_c.y[0:its]
        z = vect_c.z[0:its]
        h = vect_c.hArray[0:its]
        #det = vect_c.det[0:its]
        m = np.abs(np.array(vect_c.m[0:its]))

        if len(x) > 3 and len(y) > 3:
            r0 = np.array([x,y,z]).T
            r1 = np.array([x,y,z]).T

            r = np.array(r1[1:,:] - r0[:-1,:])
            #r = np.array([x,y]).T
            dr = 0.001
            N = r.shape[0]
            curvature = np.full(N, 0.)

            # First derivative (central difference)
            rp = np.gradient(r, dr, axis=0)
            # Second derivative (central difference)
            rpp = np.gradient(rp, dr, axis=0)

            # Numerator: determinant (x'y'' - y'x'')
            num = np.abs(rp[:, 0] * rpp[:, 1] - rp[:, 1] * rpp[:, 0])
            # Denominator: (x'^2 + y'^2)^(3/2)
            den = (rp[:,0]**2 + rp[:,1]**2)**1.5
            #print(den)
                        
            curvature = num / den + 1

        else:
            curvature = np.zeros(len(x)) + 1
        #if np.mean(curvature) < 4 and np.mean(m) < 2000:
        lc = colorline(ax, x[1:-1], y[1:-1], mag=curvature[1:-1], cmap='jet', norm = norm)

    Z = detCalc.det(lim, sep, t*np.pi, False)
    ax.imshow(Z, extent=(*[-lim, lim], *[-lim + offset_y, lim + offset_y]), origin='lower',
        aspect='equal', cmap = 'jet', norm=detCalc.detNorm)
    Z=[]

    #ax.set_ylim(-1,0)
    #ax.set_xlim(-.5,.5)
    ax.set_xlim(-.2,.2)
    ax.set_ylim(-1.7,-1.3)

    title = 'pi* ' + str(int(((t))*100)/100) +' S:' + str(int(sep*10) / 10)
    ax.set_title(title)
    artists.append([ax])

    return [ax]
    
# Start plot
fig = plt.figure(1)
ax = fig.add_subplot()#projection='3d')
fig.suptitle('Integral curve colored with curvature over heatmap of eigenvalues')
fig.colorbar(cm.ScalarMappable(cmap='jet', norm = detCalc.detNorm), ax=ax, label = 'Determinant')

# Setup animation
title = 'IterSeps'
ani = animation.FuncAnimation(fig, partial(frame, totals=totals, t = .5, sep = 1.5, detCalc = detCalc, lim = lim), frames=frame_num)
ani.save('plots/ani'+title+'.gif', writer='pillow', fps=2)
t = np.linspace(0,1,frame_num)
#plt.show()


import pickle
with open('data', 'wb') as f:
    pickle.dump(totals, f)


print(title + ': ' + str(dt.now() - start))