import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import force_calc, load_coefs, force_x_calc, force_y_calc, force_z_calc
import scipy.optimize as sp

def f(x,a,b,c):
    return a * x**b + c

def f2(x,a,c):
    return a* (1 - 1/(x**c))

def make_csvs():
    import subprocess
    power = 1
    R = 1
    for power in [1]:
        for omega in [1]:
            folder = f"cppScripts/csvs/xyz_power_{power}_omega_{omega}"
            for l in np.arange(2,201,1):
                #subprocess.run(["/Users/cdavis/Desktop/Work/Research/TendexVortexPlotting/12/cppScripts/nlp_xyz_optimized", str(l), str(0), str(power), str(R), str(omega), folder, str(500), str(500), str(500), str(1), str(1)])
                subprocess.run(["/Users/cdavis/Desktop/Work/Research/TendexVortexPlotting/12/cppScripts/overlap", 'cppScripts/params_z.txt', str(l)])
                


def lmax_plot(ax, omega, lMax, constraint='power', fit=True, t='_xyz', metric='z'):
    ls = []
    forces = []
    lMaxs = np.arange(2,lMax,1)
    for lMax in lMaxs:
        # Historical naming differs between z-only and xyz output folders.
        if t == '_xyz':
            coef_path = f'cppScripts/csvs/{constraint}_omega_{omega}{t}/maximized_coefs{t}_lMax_{lMax}_Power_1_{constraint}.csv'
        else:
            coef_path = f'cppScripts/csvs/{constraint}_0.5_omega_{omega}/maximized_coefs_lMax_{lMax}_Power_0_{constraint}.csv'

        A, B = load_coefs(coef_path, norm=False)

        if metric == 'z':
            force = force_z_calc(lMax, 2, A, B, omega)
        elif metric == 'x':
            force = force_x_calc(lMax, 2, A, B, omega)
        elif metric == 'y':
            force = force_y_calc(lMax, 2, A, B, omega)
        elif metric == 'mag':
            force = force_calc(lMax, 2, A, B, omega)
        else:
            raise ValueError("metric must be one of: 'z', 'x', 'y', 'mag'")

        ls += [lMax]
        forces += [force]

    print(f"Omega: {omega}. Max force: {np.max(forces)}")

    ax.plot(ls, forces, label='Calculated', c='orange', lw=2)

    if fit:
        ls = np.array(ls)
        forces = np.array(forces)

        popt, pcov = sp.curve_fit(f2, ls, forces, p0 =(10, 1.6))


        a = int(popt[0] * 100) / 100
        c = int(popt[1] * 100) / 100
        ax.plot(ls, f2(ls, popt[0], popt[1]), label='Fit', ls=':', c='g', lw=4)
        #ax.set_title(f'Omega: {omega} Fit: $1 - 1/{{\ell}}^{{{c}}}$ Power Constrained to $P = 0.5$')

    else:
        ax.set_title(f'Omega: {omega}')
    
    ax.set_xlabel(r'$\ell_{\text{max}}$')
    if metric == 'mag':
        ax.set_ylabel(r'$|F|$')
    elif metric == 'x':
        ax.set_ylabel(r'$F_x$')
    elif metric == 'y':
        ax.set_ylabel(r'$F_y$')
    else:
        ax.set_ylabel(r'$F_z$')

    ax.axhline(.5, c='black', ls='--')
    
def magnitude_plot(ax, constraint = 'power', lMax=200):
    
    for omega in [.7, .8, .9, 1, 1.1, 1.2, 1.3]:
        A, B = load_coefs(f'csvs/{constraint}_1_omega_{omega}/maximized_coefs_lMax_{lMax}_Power_1_{constraint}.csv', norm=False)

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
    
def force_plot(ax, constraint = 'power', lMax=200):
    
    for omega in [1]:# [.7, .8, .9, 1, 1.1, 1.2, 1.3]:
        A, B = load_coefs(f'cppScripts/csvs/{constraint}_1_omega_{omega}/maximized_coefs_lMax_{lMax}_Power_1_{constraint}.csv', norm=False)

        force_list = []
        for l in range(2,lMax+1):
            force_list += [force_calc(l, l-1, A, B, omega)]

        ax.plot(force_list, c='black')#, label=omega)

    ax.set_ylabel(r'$F_z$ Contribution per Multipole Order')
    ax.set_xlabel(r'$\ell$ Value')

def force_by_name(ax, name, lMax, omega=1):
    A, B = load_coefs(name, norm=False)

    force_list = []
    for l in range(2,lMax+1):
        force_list += [force_calc(l, l-1, A, B, omega)]
    print('Max: ', force_calc(lMax, 2, A, B, omega))
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

def lmax_plot_xyz(ax, omega, lMax, constraint = 'power', fit = True, t = '_xyz'):
    ls = []
    Fxs = []
    Fys = []
    Fzs = []
    lMaxs = np.arange(2,lMax,1)
    temp = ''
    if constraint == 'power':
        temp = '_1'
    for lMax in lMaxs:
        A, B = load_coefs(f'cppScripts/csvs/{constraint}_omega_{omega}{t}/maximized_coefs{t}_lMax_{lMax}_Power_1_{constraint}.csv', norm=False)

        Fx = force_x_calc(lMax, 2, A, B, omega)
        Fy = force_y_calc(lMax, 2, A, B, omega)
        Fz = force_z_calc(lMax, 2, A, B, omega)
        ls += [lMax]
        Fxs += [Fx]
        Fys += [Fy]
        Fzs += [Fz]

    print(f"Omega: {omega}. Max force: {np.max([Fx, Fy, Fz])}")

    ax.plot(ls, Fxs, label='Calculated Fx', lw=2)
    ax.plot(ls, Fys, label='Calculated Fy', lw=2)
    ax.plot(ls, Fzs, label='Calculated Fz', lw=2)

    if fit:
        ls = np.array(ls)
        forces = np.array(forces)

        popt, pcov = sp.curve_fit(f2, ls, forces, p0 =(10, 1.6))


        a = int(popt[0] * 100) / 100
        c = int(popt[1] * 100) / 100
        ax.plot(ls, f2(ls, popt[0], popt[1]), label='Fit', ls=':', c='g', lw=4)
        #ax.set_title(f'Omega: {omega} Fit: $1 - 1/{{\ell}}^{{{c}}}$ Power Constrained to $P = 1$')

    else:
        #fig.set_title(f'Omega: {omega}')
        pass
    ax.set_xlabel(r'$\ell_{\text{max}}$')
    ax.set_ylabel(r'$F_z$')

    ax.axhline(1, c='black', ls='--')

def plot_convergence(ax, folder, lMax):
    import pandas as pd
    import glob
    import os
    path = folder # use your path
    all_files = glob.glob(os.path.join(path , "*.csv"))

    forces = np.zeros(lMax-2)
    bounds = np.zeros(lMax-2)
    ls = np.arange(2, lMax, 1)

    for filename in all_files:
        l = int(filename.split('/')[-1].split('_')[4])
        if l < lMax:
            df = pd.read_csv(filename, header=0)
            bound_tight = df.iloc[17,1]
            force = df.iloc[8, 1]

            bounds[l - 2] = bound_tight
            forces[l - 2] = force
    ax.plot(ls, forces, label='forces')
    ax.plot(ls, bounds, label='bounds')


lMaxs = []
forces = []
lMax = 19
fig, ax = plt.subplots(nrows=1, ncols=1)

#magnitude_plot(ax[0])
#force_plot(ax, lMax=200)
#lmax_plot(ax, 1, 200, 'power', True, t='_xyzf', metric='z')
#lmax_plot_xyz(ax, 1, 9, constraint = 'power', fit = False, t = '_xyz')

make_csvs()
#plot_convergence(ax, r'/Users/cdavis/Desktop/Work/Research/TendexVortexPlotting/12/cppScripts/csvs/xyz_power_1_omega_1', lMax)
#force_by_name(ax, f'cppScripts/tmp_csvs_sched_10h/maximized_coefs_xyz_lMax_{lMax}_Power_1_power.csv', lMax)
#ax.legend()
#fig.suptitle(r'$F_z$ as it varys by max $\ell$ and $\Omega$ | $\ell_{\text{max}} = %s$ | $d\theta$ = 0.1' % (lMax))
#fig.suptitle(r'Force Observed in System Maximized up to $\ell_{\text{max}}$ Value')
#ig.suptitle(r'Coeffecient Magnitude and Force per $\ell$ value. Constrained by power $P = 1$')
plt.show()

