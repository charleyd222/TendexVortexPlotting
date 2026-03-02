import numpy as np
from scipy.special import roots_legendre
from math import pi
import quaternionic
import spherical
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import csv

# Parameters
lMax = 10 # Maximum multipole moment
threshold = 1e-12 # Minimum coeffecient to not be assumed to be 0
lam = 1.0 # Multiplicative increase to E0
dtheta = .01 # Size of gaussian in E0

# resolution
N_theta = 100
N_phi = 100

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

def E0_phi(theta_idx, phi_idx):
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
    T *= lam * exp_factor_theta_phi[theta_idx, phi_idx]
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
exp_factor_theta = np.exp(-(theta_nodes**2) / (2.0 * (dtheta**2)))  # shape (N_theta,)

theta_term = (theta_nodes[:, None] - np.pi/2)**2      # shape (N_theta, 1)
phi_term   = (phi_nodes[None, :] - np.pi)**2          # shape (1, N_phi)
exp_factor_theta_phi = np.exp(-(theta_term + phi_term) / (2.0 * (dtheta**2)))  # shape (N_theta,)



# Make E0 matrix
E0_mats = np.empty((N_theta, N_phi, 3, 3), dtype=complex)

for k in range(N_theta):
    for j in range(N_phi):
        E0_mats[k, j] = E0(k, j)

#evals = np.linalg.eigvals(E0_mats.real)

#plt.imshow(np.max(evals, axis=2))
#plt.show()
#1/0
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

    save_coefs(M, L, A_re_mat, A_im_mat, B_re_mat, B_im_mat)

    return M, L, A_re_mat, A_im_mat, B_re_mat, B_im_mat

def pm2_m_check(ell_min, ell_max):
    A_re_mat = np.full((ell_max, 2 * ell_max+1), fill_value=np.nan)
    B_re_mat = np.full((ell_max, 2 * ell_max+1), fill_value=np.nan)
    A_im_mat = np.full((ell_max, 2 * ell_max+1), fill_value=np.nan)
    B_im_mat = np.full((ell_max, 2 * ell_max+1), fill_value=np.nan)

    wigner = spherical.Wigner(ell_max, 2)

    for ell in range(ell_min, ell_max+1):
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

    save_coefs(M, L, A_re_mat, A_im_mat, B_re_mat, B_im_mat)

    return M, L, A_re_mat, A_im_mat, B_re_mat, B_im_mat

def reprocess(A_re, A_im, B_re, B_im, threshold = 1e-8):
    A_re_list = ["A_re_V"]
    A_im_list = ["A_im_V"]
    B_re_list = ["B_re_V"]
    B_im_list = ["B_im_V"]
    m_list = ["M"]
    l_list = ["L"]

    l_length = A_re.shape[0]
    m_length = A_re.shape[1]
    for l in range(l_length):
        for m in range(m_length):
            got = False
            A_re_T = 0
            A_im_T = 0
            B_re_T = 0
            B_im_T = 0

            
            if not np.isnan(A_re[l,m]) and np.abs(A_re[l,m]) > threshold:
                got = True
                A_re_T = A_re[l,m]

            if not np.isnan(A_im[l,m]) and np.abs(A_im[l,m]) > threshold:
                got = True
                A_im_T = A_im[l,m]

            if not np.isnan(B_re[l,m]) and np.abs(B_re[l,m]) > threshold:
                got = True
                B_re_T = B_re[l,m]

            if not np.isnan(B_im[l,m]) and np.abs(B_im[l,m]) > threshold:
                got = True
                B_im_T = B_im[l,m]

            if got:
                #print(l+1, m - l_length, A_re_T, B_im_T)
                
                A_re_list += [A_re_T]
                A_im_list += [A_im_T]
                B_re_list += [B_re_T]
                B_im_list += [B_im_T]
                l_list += [l+1]
                m_list += [m - l_length]

    return A_re_list, A_im_list, B_re_list, B_im_list, m_list, l_list

def load_csv(gauss_dtheta, ell_max, name=None):
    import pandas as pd
    if name== None:
        df = pd.read_csv(f'data_gauss_{str(gauss_dtheta).replace(".", "p")}_lMax_{ell_max}.csv', header=None, index_col=0)
    else:
        df = pd.read_csv(name, header=None, index_col=0)
    A_re = np.array(df.iloc[0].tolist())
    A_im = np.array(df.iloc[1].tolist())
    B_re = np.array(df.iloc[2].tolist())
    B_im = np.array(df.iloc[3].tolist())
    M = np.array(df.iloc[4].tolist(), dtype='int')
    L = np.array(df.iloc[5].tolist(), dtype='int')

    A_re_mat = np.full((ell_max, 2 * ell_max+1), fill_value=np.nan)
    B_re_mat = np.full((ell_max, 2 * ell_max+1), fill_value=np.nan)
    A_im_mat = np.full((ell_max, 2 * ell_max+1), fill_value=np.nan)
    B_im_mat = np.full((ell_max, 2 * ell_max+1), fill_value=np.nan)

    for ell in range(2, ell_max+1):
        ell_indices = np.where(L == ell)[0]
        A_re_mat[ell-1, ell_max - ell : ell_max - ell + (2 * ell + 1)] = 0
        A_im_mat[ell-1, ell_max - ell : ell_max - ell + (2 * ell + 1)] = 0
        B_re_mat[ell-1, ell_max - ell : ell_max - ell + (2 * ell + 1)] = 0
        B_im_mat[ell-1, ell_max - ell : ell_max - ell + (2 * ell + 1)] = 0

        for mI in range(0, 2*ell+1):
            m = mI-ell

            m_indices = np.where(M == m)[0]
            index = np.intersect1d(ell_indices, m_indices)

            try:
                index = index[0]

                A_re_mat[ell-1, ell_max + m] = A_re[index]
                A_im_mat[ell-1, ell_max + m] = A_im[index]
                B_re_mat[ell-1, ell_max + m] = B_re[index]
                B_im_mat[ell-1, ell_max + m] = B_im[index]
            except:
                pass
    M = np.linspace(-ell_max, ell_max, 2*ell_max+1)
    L = np.linspace(2,ell_max, ell_max-1)
    return M, L, A_re_mat, A_im_mat, B_re_mat, B_im_mat

def save_coefs(M, L, A_re_mat, A_im_mat, B_re_mat, B_im_mat):
    A_re_list, A_im_list, B_re_list, B_im_list, m_list, l_list = reprocess(A_re_mat, A_im_mat, B_re_mat, B_im_mat)

    title = 'data_gauss_' + str(dtheta).replace(".","p") + '_lMax_' + str(lMax) +  ".csv"
    with open(title, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        writer.writerow(A_re_list)
        writer.writerow(A_im_list)
        writer.writerow(B_re_list)
        writer.writerow(B_im_list)
        writer.writerow(m_list)
        writer.writerow(l_list)

fig, ax = plt.subplots(2,2)

#M, L, A_re_mat, A_im_mat, B_re_mat, B_im_mat = all_m_check(30)
#M, L, A_re_mat, A_im_mat, B_re_mat, B_im_mat = pm2_m_check(2, lMax)
M, L, A_re_mat, A_im_mat, B_re_mat, B_im_mat = load_csv(0.001, 100, name='data_gauss_y_0p001_lMax_100.csv')

print(A_re_mat)

m = np.nanmax([A_re_mat, np.real(A_im_mat), B_re_mat, np.real(B_im_mat)])
norm = colors.Normalize(-m,m)

ax[0,0].set_title('A coef Real part')
ax[0,0].pcolormesh(M, L, A_re_mat[1:], shading='nearest', norm=norm, cmap='jet')
ax[0,0].set_xlabel('m')
ax[0,0].set_ylabel(r'$\ell$')
ax[0,1].set_title('A coef Imag part')
ax[0,1].pcolormesh(M, L, A_im_mat[1:], shading='nearest', norm=norm, cmap='jet')
ax[0,1].set_xlabel('m')
ax[0,1].set_ylabel(r'$\ell$')
ax[1,0].set_title('B coef Real part')
ax[1,0].pcolormesh(M, L, B_re_mat[1:], shading='nearest', norm=norm, cmap='jet')
ax[1,0].set_xlabel('m')
ax[1,0].set_ylabel(r'$\ell$')
ax[1,1].set_title('B coef Imag part')
m = ax[1,1].pcolormesh(M, L, B_im_mat[1:], shading='nearest', norm=norm, cmap='jet')
ax[1,1].set_xlabel('m')
ax[1,1].set_ylabel(r'$\ell$')

fig.suptitle(r'Coeffecients for $E_0$ with $d\theta = %s$' % (dtheta))

plt.colorbar(m, ax = ax[:,1])
plt.show()