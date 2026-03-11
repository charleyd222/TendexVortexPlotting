import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import csv

# Helper

def plot(M, L, A_re, A_im, B_re, B_im, threshold):
    ell_max = np.max(L)

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
    mag = np.nanmax([A_re_mat, A_im_mat, B_re_mat, B_im_mat])
    norm = colors.SymLogNorm(vmin=-mag,vmax=mag, linthresh=threshold)

    fig, ax = plt.subplots(2,2)

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

    plt.colorbar(m, ax = ax[:,1])

    fig.suptitle(r'Maximized Coeffecients')
    return M, L, A_re_mat, A_im_mat, B_re_mat, B_im_mat

def save_coefs(M, L, A_re, A_im, B_re, B_im):
    
    
    title = f"maximized_coefs_lMax_{np.max(L)}.csv"
    with open(title, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        writer.writerow(["A_re"]+A_re)
        writer.writerow(["A_im"]+A_im)
        writer.writerow(["B_re"]+B_re)
        writer.writerow(["B_im"]+B_im)
        writer.writerow(["M"]+list(M))
        writer.writerow(["L"]+list(L))

# OPT

def build_index_map(l_max):
    """
    Map each (l, m, 'A'/'B') to a row/column index in c.
    Ordering: for each l from 2..l_max, for each m from -l..l,
    A^{lm} then B^{lm}.
    """
    index_map = {}
    idx = 0
    for l in range(2, l_max + 1):
        for m in range(-l, l + 1):
            index_map[(l, m, 'A')] = idx;  idx += 1
            index_map[(l, m, 'B')] = idx;  idx += 1
    N = idx
    return index_map, N

def build_M(l_max, r=1.0, Omega=1.0):
    """
    Assemble the Hermitian matrix M such that dP_z/dt = c† M c.

    The three contributions from the original expression are:
      (1) alpha_lm * Re( conj(A^lm) * A^{l+1,m} )   [AA coupling]
      (2) alpha_lm * Re( conj(B^lm) * B^{l+1,m} )   [BB coupling]
      (3) Re( beta_lm * conj(A^lm) * B^{lm}  )       [AB coupling]

    where:
      alpha_lm = a * sqrt(2*(l-m+1)*(l+m+1)) * (8*r^2 / Omega^2)
      beta_lm  = (-i*m / (8*pi*l*(l+1)))     * (8*r^2 / Omega^2)

    For a Hermitian quadratic form, a cross term Re(K * conj(ci) * cj) with i≠j
    is represented by setting M[i,j] = K/2 and M[j,i] = conj(K)/2.
    """
    index_map, N = build_index_map(l_max)
    M = np.zeros((N, N), dtype=complex)

    prefactor = 8.0 * r**2 / Omega**2

    for l in range(2, l_max + 1):
        for m in range(-l, l + 1):

            # ----------------------------------------------------------------
            # Terms (1) & (2): l <-> l+1 coupling via alpha_lm
            # Only exists if l+1 <= l_max (and |m| <= l+1 is automatic
            # since |m| <= l < l+1)
            # ----------------------------------------------------------------
            if l + 1 <= l_max:
                a = 1/(32 * np.pi * (l+1)) * ((2 * (l-1) * (l+3)) / ((2 * l + 1) * (2 * l + 3)))**(.5)
                alpha = a * np.sqrt(2.0 * (l - m + 1) * (l + m + 1)) * prefactor

                iA_l  = index_map[(l, m, 'A')]
                iA_l1 = index_map[(l+1, m, 'A')]
                iB_l  = index_map[(l, m, 'B')]
                iB_l1 = index_map[(l+1, m, 'B')]

                # alpha is real, so M[i,j] = M[j,i] = alpha/2
                M[iA_l,  iA_l1] += alpha / 2.0
                M[iA_l1, iA_l ] += alpha / 2.0

                M[iB_l,  iB_l1] += alpha / 2.0
                M[iB_l1, iB_l ] += alpha / 2.0

            # ----------------------------------------------------------------
            # Term (3): A <-> B coupling at the same (l, m) via beta_lm
            # beta is purely imaginary, so M is Hermitian but not symmetric
            # ----------------------------------------------------------------
            beta = (-1j * m / (8.0 * np.pi * l * (l + 1))) * prefactor

            iA = index_map[(l, m, 'A')]
            iB = index_map[(l, m, 'B')]

            M[iA, iB] += beta / 2.0
            M[iB, iA] += np.conj(beta) / 2.0

    return M, index_map, N

def solve_max_force(l_max, r=1.0, Omega=1.0, verbose=True, threshold=1e-8):

    M, index_map, N = build_M(l_max, r, Omega)

    print("M Built")
    print(M.shape)

    # Sanity check
    assert np.allclose(M, M.conj().T, atol=1e-12), "M is not Hermitian — check assembly!"

    # eigh returns eigenvalues in ascending order for Hermitian matrices
    eigenvalues, eigenvectors = eigh(M)

    print("Eig loaded")

    lambda_max = eigenvalues[-1]
    c_opt = eigenvectors[:, -1]   # leading eigenvector
    print(c_opt, np.linalg.norm(c_opt))

    if verbose:
        print(f"l_max = {l_max}")
        print(f"Matrix size N = {N} x {N}")
        print(f"Eigenvalue range: [{eigenvalues[0]:.4f},  {eigenvalues[-1]:.4f}]")
        print(f"\nMax force  lambda_max = {lambda_max:.6f}")
        print(f"\nOptimal coefficients c*:")
        print(f"  {'(l,m,type)':<18} {'Re(c)':<14} {'Im(c)':<14} |c|")
        print("  " + "-"*58)
        for key, idx in sorted(index_map.items(), key=lambda x: x[1]):
            l, m, t = key
            val = c_opt[idx]
            print(f"  ({l:2d},{m:+3d},{t})        "
                  f"{val.real:+.6f}  {val.imag:+.6f}  {abs(val):.6f}")
            

    A_re = []
    A_im = []
    B_re = []
    B_im = []
    L = []
    M = []
    for key, idx in sorted(index_map.items(), key=lambda x: x[1]):
        l, m, t = key
        val = c_opt[idx]
        if np.abs(val) > threshold:
            if t=='A':
                A_re += [np.real(val)]
                A_im += [np.imag(val)]
                L += [l]
                M += [m]
            elif t=='B':
                B_re += [np.real(val)]
                B_im += [np.imag(val)]

        

    L = np.array(L)
    M = np.array(M)

    plot(M, L, A_re, A_im, B_re, B_im, threshold)
    save_coefs(M, L, A_re, A_im, B_re, B_im)
    plt.show()

    return M, eigenvalues, c_opt, index_map


# ── Example runs ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for l_max in [10]:#[2, 3, 4, 6]:
        print("=" * 65)
        M, eigs, c_opt, imap = solve_max_force(l_max, r=1.0, Omega=1.0, verbose=False)
        print()