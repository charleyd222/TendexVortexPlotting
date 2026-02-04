import numpy as np
from scipy.special import roots_legendre
from math import pi
import quaternionic
import spherical
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Parameters
lMax = 100
threshold = 1e-8
lam = 1.0
dtheta = 0.1

# resolution
N_theta = 60
N_phi = 120

# Fixed vectors and projection
e1 = np.array([1.0, 0.0, 0.0])
P = np.eye(3) - np.outer(e1, e1) # P_{ij} = delta_ij - e1_i e1_j

M = np.array([0.0, 1.0, 1j])
MBar = np.array([0.0, 1.0, -1j])

# Helper functions
def TT(T):
    """TT[T] in Wolfram: P_a_j P_b_k T_jk - 1/2 P_ab (P_jk T_kj)"""
    PTPT = P @ T @ P.T
    scalar = np.trace(P @ T)
    return PTPT - 0.5 * P * scalar

def S(T):
    return 0.5 * (T + T.T)

def STT(T):
    return S(TT(T))

def Y2_Yn2(ell, m, theta, phi, wigner): # 2 and -2 Y lm
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    THf = TH.ravel()
    PHf = PH.ravel()

    # build quaternion rotations for all points
    R = quaternionic.array.from_spherical_coordinates(THf, PHf)  # length = N_theta*N_phi

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

def E0(theta_idx, phi_idx):
    ct = cos_theta[theta_idx]
    c2p = cos2phi[phi_idx]
    s2p = sin2phi[phi_idx]
    

    T = np.zeros((3,3), dtype=complex)
    T[0, :] = 0
    T[:, 0] = 0
    T[1,1] = c2p * (ct**2)
    T[1,2] = - s2p * ct
    T[2,1] = - s2p * ct
    T[2,2] = c2p
    T *= lam * exp_factor_theta[theta_idx]
    return STT(T)

# Create spectral grid
# Use Legendre for u = cos(theta) in [-1,1].
# integral over theta with sin(theta) dtheta becomes integral over u = cos(theta)

u_nodes, u_weights = roots_legendre(N_theta) # u in [-1,1]
theta_nodes = np.arccos(u_nodes) # theta in [0, pi]
# phi nodes uniform
phi_nodes = (2.0 * pi) * (np.arange(N_phi) / N_phi)
phi_weight = 2.0 * pi / N_phi # trapezoidal rule weight in phi (constant)

# Precompute commonly used trig arrays
cos_theta = np.cos(theta_nodes) # shape (N_theta,)
sin_theta = np.sin(theta_nodes)
sin_theta_half = np.sin(theta_nodes / 2.0)
cot_theta_half = np.cos(theta_nodes / 2.0) / np.where(sin_theta_half == 0, 1e-300, sin_theta_half)

cos2phi = np.cos(2.0 * phi_nodes) # shape (N_phi,)
sin2phi = np.sin(2.0 * phi_nodes)
exp_factor_theta = np.exp(-(theta_nodes**2) / (2.0 * dtheta))  # shape (N_theta,)

# Make E0 matrix
E0_mats = np.empty((N_theta, N_phi, 3, 3), dtype=complex)

for k in range(N_theta):
    for j in range(N_phi):
        E0_mats[k, j] = E0(k, j)


# Make MM and MbarMBar Matrices
MM = np.outer(M, M)
MBarMBar = np.outer(MBar, MBar)

def compute_Amode(p2Ylm, n2Ylm):

    total = 0+0j
    for k in range(N_theta):
        w_u = u_weights[k]

        Y_minus = n2Ylm[k, :] # shape (N_phi,)
        Y_plus  = p2Ylm[k, :] # shape (N_phi,)

        alpha = MM[None, :, :] * Y_minus[:, None, None]
        beta = MBarMBar[None, :, :] * Y_plus[:, None, None]

        integral = np.einsum('...ij,...ij->...', E0_mats[k], np.conjugate(alpha + beta))

        total += np.sum(integral) * (w_u * phi_weight)

    return total

def compute_Bmode(p2Ylm, n2Ylm):

    total = 0+0j
    for k in range(N_theta):
        w_u = u_weights[k]

        Y_minus = n2Ylm[k, :] # shape (N_phi,)
        Y_plus  = p2Ylm[k, :] # shape (N_phi,)

        alpha = MM[None, :, :] * Y_minus[:, None, None]
        beta = MBarMBar[None, :, :] * Y_plus[:, None, None]

        integral = np.einsum('...ij,...ij->...', E0_mats[k], np.conjugate(-1j * (alpha - beta)))

        total += np.sum(integral) * (w_u * phi_weight)

    return total

def all_m_check(ell_max):
    A_re_mat = np.full((ell_max, 2 * ell_max+1), fill_value=np.nan)
    B_re_mat = np.full((ell_max, 2 * ell_max+1), fill_value=np.nan)
    A_im_mat = np.full((ell_max, 2 * ell_max+1), fill_value=np.nan)
    B_im_mat = np.full((ell_max, 2 * ell_max+1), fill_value=np.nan)

    wigner = spherical.Wigner(ell_max, 2)

    for ell in range(2, ell_max+1):
        print(ell)
        A_re_temp = np.zeros(2 * ell+1)
        B_re_temp = np.zeros(2 * ell+1)
        A_im_temp = np.zeros(2 * ell+1)
        B_im_temp = np.zeros(2 * ell+1)
        for m in range(0, 2*ell+1):
            p2Ylm, n2Ylm = Y2_Yn2(ell, m-ell, theta_nodes, phi_nodes, wigner)

            Alm = compute_Amode(p2Ylm, n2Ylm)
            Blm = compute_Bmode(p2Ylm, n2Ylm)

            A_re_temp[m] = Alm.real
            B_re_temp[m] = Blm.real
            A_im_temp[m] = Alm.imag
            B_im_temp[m] = Blm.imag

        A_re_mat[ell-1, ell_max - ell : ell_max - ell + (2 * ell + 1)] = A_re_temp
        B_re_mat[ell-1, ell_max - ell : ell_max - ell + (2 * ell + 1)] = B_re_temp
        A_im_mat[ell-1, ell_max - ell : ell_max - ell + (2 * ell + 1)] = A_im_temp
        B_im_mat[ell-1, ell_max - ell : ell_max - ell + (2 * ell + 1)] = B_im_temp

    M = np.linspace(-ell_max, ell_max, 2*ell_max+1)
    L = np.linspace(2,ell_max, ell_max-1)

    return M, L, A_re_mat, A_im_mat, B_re_mat, B_im_mat

def pm2_m_check(ell_max):
    A_re_mat = np.full((ell_max, 2 * ell_max+1), fill_value=np.nan)
    B_re_mat = np.full((ell_max, 2 * ell_max+1), fill_value=np.nan)
    A_im_mat = np.full((ell_max, 2 * ell_max+1), fill_value=np.nan)
    B_im_mat = np.full((ell_max, 2 * ell_max+1), fill_value=np.nan)

    wigner = spherical.Wigner(ell_max, 2)

    for ell in range(2, ell_max+1):
        #print(ell)
        A_re_temp = np.zeros(2 * ell+1)
        B_re_temp = np.zeros(2 * ell+1)
        A_im_temp = np.zeros(2 * ell+1)
        B_im_temp = np.zeros(2 * ell+1)
        for m in range(0, 2*ell+1):
            if abs(m-ell) != 2:
                continue
            p2Ylm, n2Ylm = Y2_Yn2(ell, m-ell, theta_nodes, phi_nodes, wigner)

            Alm = compute_Amode(p2Ylm, n2Ylm)
            Blm = compute_Bmode(p2Ylm, n2Ylm)

            A_re_temp[m] = Alm.real
            B_re_temp[m] = Blm.real
            A_im_temp[m] = Alm.imag
            B_im_temp[m] = Blm.imag

        A_re_mat[ell-1, ell_max - ell : ell_max - ell + (2 * ell + 1)] = A_re_temp
        B_re_mat[ell-1, ell_max - ell : ell_max - ell + (2 * ell + 1)] = B_re_temp
        A_im_mat[ell-1, ell_max - ell : ell_max - ell + (2 * ell + 1)] = A_im_temp
        B_im_mat[ell-1, ell_max - ell : ell_max - ell + (2 * ell + 1)] = B_im_temp

    M = np.linspace(-ell_max, ell_max, 2*ell_max+1)
    L = np.linspace(2,ell_max, ell_max-1)

    return M, L, A_re_mat, A_im_mat, B_re_mat, B_im_mat

fig, ax = plt.subplots(2,2)

#M, L, A_re_mat, A_im_mat, B_re_mat, B_im_mat = all_m_check(5)
M, L, A_re_mat, A_im_mat, B_re_mat, B_im_mat = pm2_m_check(5)

norm = colors.Normalize(-.5,.5)

ax[0,0].set_title('A coef Real part')
ax[0,0].pcolormesh(M, L, A_re_mat[1:], shading='nearest', norm=norm, cmap='jet')
ax[0,0].set_xlabel('m')
ax[0,0].set_ylabel('l')
ax[0,1].set_title('A coef Imag part')
ax[0,1].pcolormesh(M, L, A_im_mat[1:], shading='nearest', norm=norm, cmap='jet')
ax[0,1].set_xlabel('m')
ax[0,1].set_ylabel('l')
ax[1,0].set_title('B coef Real part')
ax[1,0].pcolormesh(M, L, B_re_mat[1:], shading='nearest', norm=norm, cmap='jet')
ax[1,0].set_xlabel('m')
ax[1,0].set_ylabel('l')
ax[1,1].set_title('B coef Imag part')
m = ax[1,1].pcolormesh(M, L, B_im_mat[1:], shading='nearest', norm=norm, cmap='jet')
ax[1,1].set_xlabel('m')
ax[1,1].set_ylabel('l')
plt.colorbar(m, ax = ax[:,1])
plt.show()