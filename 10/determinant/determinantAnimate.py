from ctypes import *
from datetime import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import matplotlib.animation as animation
from functools import partial

# CHOOSE MODEL
singleQuad = False
doubleQuadSeperated = True


# PARAMS
frame_num = 11
x_range = [-5,5]
y_range = [-5,5]
n = 1000
m = 1000
sepX = 2
sepY = 2
theta = np.pi

if singleQuad:
    detCalc = CDLL("./determinantCalc").f1Quadropole
    detCalc.argtypes = [c_double, c_double]
elif doubleQuadSeperated:
    detCalc = CDLL("./determinantCalc").fCustomT2Quadropole
    detCalc.argtypes = [c_double, c_double, c_double, c_double, c_double]


detCalc.restype = c_double
norm = mcolors.SymLogNorm(1e-4, vmin=-1e5, vmax=1e5)
seps = np.linspace(.1,4,frame_num)
t_list = np.linspace(0,1,frame_num)


def frame(frame, theta=None, sep=None):
    ax.cla()

    if sep == None:
        sep = seps[frame]
    if theta == None:
        theta= t_list[frame]

    x_vals = np.linspace(x_range[0], x_range[1], n)
    y_vals = np.linspace(y_range[0], y_range[1], m)
    Z = np.zeros((m, n))

    # Compute f(x, y) point by point
    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            if singleQuad:
                Z[i, j] = detCalc(x, y)
            elif doubleQuadSeperated:
                val = detCalc(x, y, sep, sep, theta * np.pi)
                #if np.abs(val) < 1e-4:
                #    val = 1e-4
                Z[i, j] = val

    # Plot heatmap with square bins
    ax.imshow(Z, extent=(*x_range, *y_range), origin='lower',
                aspect='equal', cmap = 'jet', norm=norm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.suptitle('%s pi rotated. %s separation' % (theta, sep))

# Start plot
fig, ax = plt.subplots(1)
fig.colorbar(cm.ScalarMappable(cmap='jet', norm = norm), ax=ax, label = 'Abs Val of Determinant')

ani = animation.FuncAnimation(fig, partial(frame, theta = 0), frames=frame_num)
ani.save('sep.gif', writer='pillow', fps=2)