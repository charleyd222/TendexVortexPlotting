import matplotlib.pyplot as plt
import numpy as np
import quaternionic
import spherical

from scipy.special import binom
from math import factorial

def eigen(T):
    _, _, nY, nX = T.shape

    T_Evals = np.zeros((nY, nX))
    for y in range(nY):
        for x in range(nX):
            vals = np.linalg.eigvals(np.real(T[:,:,y, x]))
            T_Evals[y,x] = np.real(np.max(vals))
            

    return T_Evals

def E0(theta, phi, l, dTheta):
    exp = np.exp(-theta**2 / (2 * dTheta))

    a = np.cos(2 * phi) * (np.sin(theta))**2
    b = np.cos(theta) * np.sin(2 * phi)

    E = l * exp * np.array([[-a, -b], 
                        [-b, a]])
    
    return E

def sYlm(s, l, m, theta, phi):
    summation = 0

    for r in range(0,l-s+1):
        summation += binom(l-s, r) * binom(l + s, r + s - m) * (-1)**(l-r-s) * np.exp(1.j * m * phi) / (np.tan(theta / 2) ** (2 * r + s - m))
    
    Y = (-1)**m * np.sqrt(factorial(l+m) * factorial(l-m) * (2 * l + 1) / (factorial(l + s) * factorial(l - s) * 4 * np.pi)) * (np.sin(theta / 2)**(2*l)) * summation
    
    Y = np.nan_to_num(Y, 0)
    return Y

def Y2_Yn2(ell,m, theta, phi): # 2 and -2 Y lm
    THf = theta.ravel()
    PHf = phi.ravel()

    # build quaternion rotations for all points
    R = quaternionic.array.from_spherical_coordinates(THf, PHf)  # length = N_theta*N_phi

    # precompute Wigner ell_min to ell_max
    wigner = spherical.Wigner(ell, ell)

    # evaluate spin-weighted spherical harmonics for spin s at all R points
    Yn2 = wigner.sYlm(-2, R)   # vectorized evaluation (num_points x n_modes) or similar
    Y2 = wigner.sYlm(2, R)   # vectorized evaluation (num_points x n_modes) or similar

    # Access specific mode (Set of quantum numbers)
    mode_idx = wigner.Yindex(ell=ell, m=m)
    Y2_flat = Y2[:, mode_idx]
    Yn2_flat = Yn2[:, mode_idx]

    # reshape back to the grid:
    Y2_grid = Y2_flat.reshape(TH.shape)   # shape (N_theta, N_phi)
    Yn2_grid = Yn2_flat.reshape(TH.shape)   # shape (N_theta, N_phi)

    return Y2_grid, Yn2_grid

def TE2lm_calc(Y2, Yn2):
    
    a = Yn2 + Y2
    b = 1.j * (Yn2 - Y2)

    E = np.array([[a, b], 
                  [b, -a]])
    E *= 2 **-1.5

    return E

def TB2lm_calc(Y2, Yn2):
    
    a = Yn2 - Y2
    b = 1.j * (Yn2 + Y2)

    E = np.array([[a, b], 
                  [b, -a]])
    E *= -1.j/ (2 **1.5)

    return E

def I_calc(ell, m, t, r, Omega = 1, omega = 0):
    return Omega ** (ell+2) * np.exp(-1.j * Omega * ((t-r) + omega))

fig, ax = plt.subplots(3)

N_theta, N_phi = 128, 256
theta = np.linspace(0.0, np.pi, N_theta)
phi   = np.linspace(0.0, 2*np.pi, N_phi, endpoint=False)
TH, PH = np.meshgrid(theta, phi, indexing='ij')   # shape (N_theta, N_phi)

l = 2
m = 2

t = 0
R = 1
I = I_calc(l, m, t, R, omega = 0)
S = (4/3) * I_calc(l, m, t, R, omega=np.pi/2)


Y2, Yn2 = Y2_Yn2(l, m, TH, PH)

TE2 = I * TE2lm_calc(Y2, Yn2)
TB2 = S * TB2lm_calc(Y2, Yn2)

vmax = .5

ax[0].imshow(
    eigen(TE2),
    extent=[0, 2*np.pi, 0, np.pi],
    origin='lower',
    aspect='auto',
    vmin=0, vmax=vmax,
    cmap='viridis'
)
ax[0].set_title('Mass Quadropole')

m = ax[1].imshow(
    eigen(TB2),
    extent=[0, 2*np.pi, 0, np.pi],
    origin='lower',
    aspect='auto',
    vmin=0, vmax=vmax,
    cmap='viridis'
)
ax[1].set_title('Current Quadropole')

m = ax[2].imshow(
    eigen(TE2 + TB2),
    extent=[0, 2*np.pi, 0, np.pi],
    origin='lower',
    aspect='auto',
    vmin=0, vmax=vmax,
    cmap='viridis'
)
ax[2].set_title('Mass and Current Quadropole summed')
fig.colorbar(m, ax=ax[:])
fig.supxlabel('Phi')
fig.supylabel('Theta')
plt.show()