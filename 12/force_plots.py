import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import force_calc, full_force_calc
import scipy.optimize as sp

def f(x,a,b,c):
    return a * x**b + c

def load(name):
    df = pd.read_csv(name, header=None, index_col=0)

    A_re = np.array(df.iloc[0].tolist(), dtype='complex128')
    A_im = np.array(df.iloc[1].tolist(), dtype='complex128')
    B_re = np.array(df.iloc[2].tolist(), dtype='complex128')
    B_im = np.array(df.iloc[3].tolist(), dtype='complex128')
    M = np.array(df.iloc[4].tolist(), dtype='int')
    L = np.array(df.iloc[5].tolist(), dtype='int')

    A = {}
    B = {}
    for i, ml in enumerate(zip(M, L)):
        m, l = ml
        ml = (int(m), int(l))
        A[ml] = A_re[i] + (1.j * A_im[i])
        B[ml] = B_re[i] + (1.j * B_im[i])

    return A, B

def alternate_omega(ax, A, B):
    for omega in [1]:
        f_omega_y = []
        f_omega_x = []
        base = force_calc(2, omega, A, B)
        for l in range(2,lMax):
            f_omega_y += [force_calc(l, omega, A, B)]# / base]
            f_omega_x += [l]

        popt, pcov = sp.curve_fit(f, f_omega_x, f_omega_y)
        x = np.linspace(2,lMax,100)

        ax.plot(x, f(x, popt[0], popt[1], popt[2]))
        ax.scatter(f_omega_x, f_omega_y, label=omega)

        ax.set_title(r'Varying $\Omega$')
        ax.set_ylabel(r'Force increase over $\ell = 2$')
        ax.set_xlabel(r'$\ell$ Max')
        ax.legend()

def alternate_omega_full(ax1, ax2, ax3, A, B):
    for omega in [1]:
        f_omega_y1 = []
        f_omega_y2 = []
        f_omega_y3 = []
        f_omega_x = []
        base = full_force_calc(2, omega, A, B)
        print(base)
        for l in range(2,lMax):
            val = full_force_calc(l, omega, A, B)
            f_omega_y1 += [val[0] / base[0]]
            f_omega_y2 += [val[1]]# / base[1]]
            f_omega_y3 += [val[2]]# / base[2]]
            f_omega_x += [l]

        #popt, pcov = sp.curve_fit(f, f_omega_x, f_omega_y)
        #x = np.linspace(2,lMax,100)

        #ax.plot(x, f(x, popt[0], popt[1], popt[2]))
        #ax.scatter(f_omega_x, f_omega_y, label=omega)

        ax1.plot(f_omega_y1, label='x')
        ax2.plot(f_omega_y2, label='y')
        ax3.plot(f_omega_y3, label='z')

        ax1.set_title(r'Varying $\Omega$, X component')
        ax1.set_ylabel(r'Force increase over $\ell = 2$')
        ax1.set_xlabel(r'$\ell$ Max')

        ax2.set_title(r'Varying $\Omega$, Y component')
        ax2.set_ylabel(r'Force increase over $\ell = 2$')
        ax2.set_xlabel(r'$\ell$ Max')

        ax3.set_title(r'Varying $\Omega$, Z component')
        ax3.set_ylabel(r'Force increase over $\ell = 2$')
        ax3.set_xlabel(r'$\ell$ Max')

def alternate_omega_full_x(ax, A, B, ratio = False):
    for omega in [1]:
        f_omega_y = []
        f_omega_x = []
        if ratio:
            base = full_force_calc(2, omega, A, B)[0]
        else:
            base = 1

        for l in range(2,lMax):
            val = full_force_calc(l, omega, A, B)
            f_omega_y += [val[0] / base]
            f_omega_x += [l]

        #popt, pcov = sp.curve_fit(f, f_omega_x, f_omega_y)
        #x = np.linspace(2,lMax,100)

        #ax.plot(x, f(x, popt[0], popt[1], popt[2]))
        #ax.scatter(f_omega_x, f_omega_y, label=omega)

        ax.plot(f_omega_y, label='x')

        ax.set_title(r'Varying $\Omega$, X component')
        if ratio:
            ax.set_ylabel(r'Force increase over $\ell = 2$')
        else:
            ax.set_ylabel(r'$F_x$')
        ax.set_xlabel(r'$\ell$ Max')

def alternate_omega_full_y(ax, A, B, ratio = False):
    for omega in [1]:
        f_omega_y = []
        f_omega_x = []
        if ratio:
            base = full_force_calc(2, omega, A, B)[1]
        else:
            base = 1

        for l in range(2,lMax):
            val = full_force_calc(l, omega, A, B)
            f_omega_y += [val[1] / base]
            f_omega_x += [l]

        #popt, pcov = sp.curve_fit(f, f_omega_x, f_omega_y)
        #x = np.linspace(2,lMax,100)

        #ax.plot(x, f(x, popt[0], popt[1], popt[2]))
        #ax.scatter(f_omega_x, f_omega_y, label=omega)

        ax.plot(f_omega_y, label='x')

        ax.set_title(r'Varying $\Omega$, Y component')
        if ratio:
            ax.set_ylabel(r'Force increase over $\ell = 2$')
        else:
            ax.set_ylabel(r'$F_y$')
        ax.set_xlabel(r'$\ell$ Max')

def alternate_omega_full_z(ax, A, B, ratio = False):
    for omega in [1,1.1,1.2,1.3,1.4]:
        f_omega_y = []
        f_omega_x = []
        if ratio:
            base = full_force_calc(2, omega, A, B)[2]
        else:
            base = 1

        for l in range(2,lMax):
            val = full_force_calc(l, omega, A, B)
            f_omega_y += [val[2] / base]
            f_omega_x += [l]

        #popt, pcov = sp.curve_fit(f, f_omega_x, f_omega_y)
        #x = np.linspace(2,lMax,100)

        #ax.plot(x, f(x, popt[0], popt[1], popt[2]))
        #ax.scatter(f_omega_x, f_omega_y, label=omega)

        ax.plot(f_omega_y, label=omega)

        ax.set_title(r'Varying $\Omega$, Z component')
        if ratio:
            ax.set_ylabel(r'Force increase over $\ell = 2$')
        else:
            ax.set_ylabel(r'$F_z$')
        ax.set_xlabel(r'$\ell$ Max')
        ax.legend()

def alternate_l(ax, A, B, omega=1):
    for l in [3, 6, 9, 12, 15, 18]:
        f_L = []
        base = force_calc(2, omega, A, B)
        for omega in np.linspace(0.1,2,100):
            f_L += [force_calc(l, omega, A, B) / base]

        ax.plot(f_L, label=l)
        ax.set_yscale('log')
        ax.set_title(r'Varying $\ell_{\text{max}}$')
        ax.set_ylabel(r'Force increase over $\ell = 2$')
        ax.set_xlabel(r'$\Omega$ Max')
        ax.set_xticks(np.linspace(0,100,9), np.linspace(0,2,9))
        ax.legend()

lMaxs = []
forces = []
lMax = 100

A_x, B_x = load('data_gauss_x_0p001_lMax_100.csv')
A_y, B_y = load('data_gauss_y_0p001_lMax_100.csv')
A_z, B_z = load('data_gauss_z_0p001_lMax_100.csv')

fig, ax = plt.subplots(nrows=2, ncols=3)


alternate_omega_full_x(ax[0,0], A_x, A_x)
alternate_omega_full_y(ax[0,1], A_y, A_y)
alternate_omega_full_z(ax[0,2], A_z, B_z)
alternate_omega_full_x(ax[1,0], A_x, A_x, True)
alternate_omega_full_y(ax[1,1], A_y, A_y, True)
alternate_omega_full_z(ax[1,2], A_z, B_z, True)

fig.suptitle(r'$F_z$ as it varys by max $\ell$ and $\Omega$ | $\ell_{\text{max}} = %s$ | $d\theta$ = 0.001' % (lMax))
plt.show()

