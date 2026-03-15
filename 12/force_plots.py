import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import force_calc, full_force_calc, load_coefs
import scipy.optimize as sp

def f(x,a,b,c):
    return a * x**b + c

def f2(x,a,c):
    return a* (1 - 1/(x**c))

def alternate_omega(ax, A, B):
    for omega in [1]:
        f_omega_y = []
        f_omega_x = []
        base = force_calc(2, omega, A, B)
        for l in range(2,lMax):
            f_omega_y += [force_calc(l, A, B, omega)]# / base]
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

def alternate_omega_full_x(ax, A, B, ratio = False, fit = False):
    for omega in [1]:#[.6,.8,1,1.2,1,1.4]:
        f_omega_y = []
        f_omega_x = []
        if ratio:
            base = full_force_calc(2, omega, A, B)[0]
        else:
            base = 1

        for l in range(2,lMax):
            #print(full_force_calc(l, omega, A, B))
            val = full_force_calc(l, omega, A, B)
            f_omega_y += [val[0] / base]
            f_omega_x += [l]

        print(f'Maximum force: {np.max(f_omega_y)}')

        if fit:#not ratio:
            popt, pcov = sp.curve_fit(f, f_omega_x, f_omega_y)
            x = np.linspace(2,lMax,100)

            ax.plot(x, f(x, popt[0], popt[1], popt[2]))
            ax.scatter(f_omega_x, f_omega_y, label=omega)

            print(popt)
        
        ax.plot(f_omega_y, label=omega)
        
        if ratio:
            ax.set_title(r'Force increase over $\ell = 2$')#. Fit Coef: $A x^{B} + C$. A:' + popt[0] + f'. B:{popt[1]}. C:{popt[2]}')
            ax.set_ylabel(r'Force increase over $\ell = 2$')
        else:
            ax.set_title(r'Varying $\Omega$, X component')
            ax.set_ylabel(r'$F_x$')
            ax.legend()
        ax.set_xlabel(r'$\ell$ Max')

def alternate_omega_full_y(ax, A, B, ratio = False):
    for omega in [1,1.1,1.2,1.3,1.4]:
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
    for omega in [1]:
        f_omega_y = []
        f_omega_x = []
        if ratio:
            base = omega**4 / (24 * np.pi) #full_force_calc(2, omega, A, B)[2]
        else:
            base = 1

        for l in range(2,lMax+1):
            #print(l)
            val = full_force_calc(l, omega, A, B)
            f_omega_y += [val[2] / base]
            f_omega_x += [l]

        #popt, pcov = sp.curve_fit(f, f_omega_x, f_omega_y)
        #x = np.linspace(2,lMax,100)

        #ax.plot(x, f(x, popt[0], popt[1], popt[2]))
        #ax.scatter(f_omega_x, f_omega_y, label=omega)

        ax.plot(f_omega_y, label=omega)

        print(f'Maximum force: {np.max(f_omega_y)}')

        #print(popt)

        #ax.set_title(r'Varying $\Omega$, Z component. Fit Coef: $A x^{B} + C$. A:' + popt[0] + f'. B:{popt[1]}. C:{popt[2]}')
        if ratio:
            ax.set_title(r'Force increase over $\ell = 2$')
            ax.set_ylabel(r'Force increase over $\ell = 2$')
        else:
            ax.set_ylabel(r'$F_z$')
            ax.set_title(r'Varying $\Omega$, Z component')
            ax.legend()
        ax.set_xlabel(r'$\ell$ Max')     

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

def make_csvs():
    import subprocess
    power = 1
    R = 1
    for omega in [.7,.8,.9,1,1.1,1.2,1.3]:
        folder = f"csvs/power_{power}_omega_{omega}"
        for l in [201]:#np.arange(101,200,1):
            subprocess.run(["/Users/cdavis/Desktop/Work/Research/TendexVortexPlotting/12/cppScripts/nlp", str(l), str(0), str(power), str(R), str(omega), folder])

def lmax_plot(ax, omega, constraint = 'power'):
    ls = []
    forces = []
    lMaxs = np.arange(2,100,1)
    for lMax in lMaxs:
        A, B = load_coefs(f'csvs/power_1_omega_{omega}/maximized_coefs_lMax_{lMax}_Power_1_{constraint}.csv', norm=False)#, offset=1/(9.6e-5))

        force = force_calc(lMax, A, B, omega)
        ls += [lMax]
        forces += [force]

    print(f"Omega: {omega}. Max force: {np.max(forces)}")

    ls = np.array(ls)
    forces = np.array(forces)

    popt, pcov = sp.curve_fit(f2, ls, forces, p0 =(10, 1.6))


    a = int(popt[0] * 100) / 100
    c = int(popt[1] * 100) / 100
    ax.plot(ls, f2(ls, popt[0], popt[1]), label='fit')
    ax.plot(ls, forces, label='obs')

    ax.set_title(f'Omega: {omega}. {a} * (1 - 1/(l**{c}))')

def magnitude_plot(ax, omega, constraint = 'power', lMax=200):
    
    for omega in [.7, .8, .9, 1, 1.1, 1.2, 1.3]:
        A, B = load_coefs(f'csvs/power_1_omega_{omega}/maximized_coefs_lMax_{lMax}_Power_1_{constraint}.csv', norm=False)

        mags = {}

        for k in A.keys():
            l = k[1]
            mags[l] = A[k] + B[k]

        mags_list = []
        for l in range(2,lMax+1):
            mags_list += [np.abs(mags[l])]

        ax.plot(mags_list, label=omega)
    #ax.set_ylim([0,.5])
    ax.set_ylabel(r'Coeffiecient Magnitude $(|A^{\ell \, m} + B^{\ell \, m}|)$')
    ax.set_xlabel(r'$\ell$ Value')
    
def force_plot(ax, omega, constraint = 'power', lMax=200):
    
    for omega in [.7, .8, .9, 1, 1.1, 1.2, 1.3]:
        A, B = load_coefs(f'csvs/power_1_omega_{omega}/maximized_coefs_lMax_{lMax}_Power_1_{constraint}.csv', norm=False)

        force_list = []
        for l in range(2,lMax+1):
            force_list += [force_calc(l, l-1, A, B, omega)]

        ax.plot(force_list, c='black')#, label=omega)

    ax.set_ylabel(r'$F_z$ Contribution per Multipole Order')
    ax.set_xlabel(r'$\ell$ Value')

def test(A,B):
    keys = A.keys()

    As = 0
    Bs = 0
    y = []

    for k in keys:
        As += A[k] * np.conj(A[k])
        Bs += B[k] * np.conj(B[k])

        print(A[k] * np.conj(A[k]) + B[k] * np.conj(B[k]))
        y += [np.real(A[k] * np.conj(A[k]) + B[k] * np.conj(B[k]))]
    print(As, Bs)

    plt.plot(y)

lMaxs = []
forces = []
lMax = 100

#A_x, B_x = load_coefs('data_gauss_x_0p01_lMax_100.csv')
#A_y, B_y = load_coefs('data_gauss_y_0p001_lMax_100.csv')
#A_z, B_z = load_coefs('./csvs/maximized_coefs_lMax_100_Power_1_norm_per_ell_norm.csv', norm=False)
#A_z, B_z = load_coefs(f'maximized_coefs_lMax_{lMax}_norm.csv', norm=False)#, offset=1/(9.6e-5))
#A_z2, B_z2 = load_coefs('maximized_coefs_lMax_100.csv', norm=True)#, offset=1/(9.6e-5))
fig, ax = plt.subplots(nrows=1, ncols=2)
#magnitude_plot(ax[0, 0], .7)
#magnitude_plot(ax[0, 1], .8)
#magnitude_plot(ax[0, 2], .9)
#magnitude_plot(ax[1, 0], 1)
#magnitude_plot(ax[1, 1], 1.1)
magnitude_plot(ax[0], 1.2)
force_plot(ax[1], 1.2)

#A, B = load_coefs('csvs/maximized_coefs_lMax_100_norm.csv')
#make_csvs()

ax[0].legend()
#fig.suptitle(r'$F_z$ as it varys by max $\ell$ and $\Omega$ | $\ell_{\text{max}} = %s$ | $d\theta$ = 0.1' % (lMax))
fig.suptitle(r'Coeffecient Magnitude and Force per $\ell$ value. Constrained by power $P = 1$')
plt.show()

