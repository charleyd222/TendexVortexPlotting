import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import force_calc, full_force_calc, load_coefs
import scipy.optimize as sp

def f(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

A_z, B_z = load_coefs('data_gauss_z_0p001_lMax_1000.csv')

lMax = 1000

Ap2 = []
Am2 = []
Bp2 = []
Bm2 = []
l = []

for i in range(2,lMax+1):
    Ap2 += [A_z[(2, i)].real]
    Am2 += [A_z[(-2, i)].real]
    Bp2 += [B_z[(2, i)].imag]
    Bm2 += [B_z[(-2, i)].imag]

    l += [i]


fig, ax = plt.subplots(nrows=2, ncols=2)
#print(np.sum([Ap2, Am2, Bp2, Bm2]))
ax[0,0].plot(Ap2)
ax[0,0].set_title('Ap2')

#popt, pcov = sp.curve_fit(f, Ap2, l, p0 = [1e-5, 1e-5, 1e-5, 500])
#x = np.linspace(2,lMax,1000)

#ax[0,0].plot(x, f(x, popt[0], popt[1], popt[2], popt[3]))

ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')

ax[0,1].plot(Am2)
ax[0,1].set_title('Am2')

ax[1,0].plot(Bp2)
ax[1,0].set_title('Bp2')

ax[1,1].plot(Bm2)
ax[1,1].set_title('Bm2')

plt.show()
