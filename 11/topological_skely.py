import matplotlib.pyplot as plt
import numpy as np
from ctypes import *
from util import mat, vis_params
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

def find_degenerate(R, N, modeling_params, threshold=1):
    N_th, N_ph = 2*N, N

    T = CDLL("./cppScripts/rka_iter").test_val
    T.argtypes = [c_double, c_double, c_double, vis_params, c_bool]
    T.restype = mat

    theta = np.linspace(0, np.pi, N_th)
    phi = np.linspace(0, 6, N_ph)
    Phi, Theta = np.meshgrid(phi, theta)

    f = np.zeros((N_th, N_ph))  # T11 - T22
    g = np.zeros((N_th, N_ph))  # T12

    for i in range(Theta.shape[0]):
        for j in range(Theta.shape[1]):
            th = Theta[i, j]
            ph = Phi[i, j]
            
            m = T(R, th, ph, modeling_params, True)

            f[i,j] = m.m11 - m.m22
            g[i,j] = m.m12
    deg_points = f*g

    # Boolean mask of "dark" pixels
    mask = deg_points < threshold

    # Label connected components
    labels, num = ndi.label(mask)
    print(labels, num)

    # Compute centers of mass for each component
    centers = ndi.center_of_mass(mask)

    return centers, labels, f, g
