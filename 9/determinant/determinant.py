from ctypes import *
from datetime import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

class determinant():
    def __init__(self):
        self.detCalc = CDLL("./determinant/determinantCalc").fCustomT2Quadropole
        self.detCalc.argtypes = [c_double, c_double, c_double, c_double, c_double]
        self.detCalc.restype = c_double
        self.detNorm = mcolors.SymLogNorm(1e-4, vmin=-1e5, vmax=1e5)
        self.detNormAbs = mcolors.LogNorm(vmin=1e-10, vmax=1e4)

    def det(self, lim, sep, theta, n=1000):
        Z = np.zeros((n, n))
        x_vals = np.linspace(-lim, lim, n)
        y_vals = np.linspace(-lim, lim, n)
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                val = self.detCalc(x, y, sep, sep, theta)
                Z[i, j] = np.abs(val)

        return Z
        
    def single_det(self, x,y,z, sepX, sepY, theta):
        return self.detCalc(x,y,z, sepX, sepY, theta)
