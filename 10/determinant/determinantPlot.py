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
x_range = [-5,5]
y_range = [-5,5]
n = 1000
m = 1000
sepX = 1
sepY = 0
theta = 0

if singleQuad:
    detCalc = CDLL("./determinantCalc").f1Quadropole
    detCalc.argtypes = [c_double, c_double]
elif doubleQuadSeperated:
    detCalc = CDLL("./determinantCalc").fCustomT2Quadropole
    detCalc.argtypes = [c_double, c_double, c_double, c_double, c_double]


detCalc.restype = c_double
norm = mcolors.SymLogNorm(1e-4, vmin=-1e5, vmax=1e5)


x_vals = np.linspace(x_range[0], x_range[1], n)
y_vals = np.linspace(y_range[0], y_range[1], m)
Z = np.zeros((m, n))

# Compute f(x, y) point by point
for i, y in enumerate(y_vals):
    for j, x in enumerate(x_vals):
        if singleQuad:
            Z[i, j] = detCalc(x, y)
        elif doubleQuadSeperated:
            val = detCalc(x, y, sepX, sepY, theta*np.pi)
            #if np.abs(val) < 1e-4:
            #    val = 1e-4
            Z[i, j] = val
        

# Plot heatmap with square bins
plt.figure(figsize=(6, 6))
plt.imshow(Z, extent=(*x_range, *y_range), origin='lower',
            aspect='equal', cmap = 'PiYG', norm=norm)
plt.colorbar(label='Determinant')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Heatmap of Determinant, signed. Theta: %s Pi. Seperation: %s' % (theta, sepX))
plt.show()