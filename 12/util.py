from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def make_square(cx, cy, h, nx, ny):
    # --- 1. Construct square boundary (counterclockwise) ---
    pts = []

    # bottom edge (left -> right)
    xs = np.linspace(cx - h, cx + h, nx, endpoint=False)
    for x in xs:
        pts.append([x, cy - h])

    # right edge (bottom -> top)
    ys = np.linspace(cy - h, cy + h, ny, endpoint=False)
    for y in ys:
        pts.append([cx + h, y])

    # top edge (right -> left)
    xs = np.linspace(cx + h, cx - h, nx, endpoint=False)
    for x in xs:
        pts.append([x, cy + h])

    # left edge (top -> bottom)
    ys = np.linspace(cy + h, cy - h, ny, endpoint=False)
    for y in ys:
        pts.append([cx - h, y])

    return np.array(pts)

def find_singularity_direction(F, center, half_width, n_per_side=16):
    """
    Direction toward a nearby singularity using a square probing surface.

    F          : undirected 2D vector field, F(x) -> R^2
    center     : (x, y)
    half_width : half side length of square
    n_per_side : samples per side
    """

    cx, cy = center
    h = half_width

    pts = make_square(cx, cy, h, n_per_side, n_per_side)

    # --- 2. Inward normals for each boundary segment ---
    normals = []

    normals += [[0,  1]] * n_per_side   # bottom edge
    normals += [[-1, 0]] * n_per_side   # right edge
    normals += [[0, -1]] * n_per_side   # top edge
    normals += [[1,  0]] * n_per_side   # left edge

    normals = np.array(normals)

    # --- 3. Evaluate and normalize field ---
    V = np.array([F(p[0],p[1]) for p in pts])
    print(V.shape)
    V /= np.linalg.norm(V, axis=1)[:, None]

    # --- 4. Resolve ± ambiguity incrementally ---
    Vc = [V[0]]
    for i in range(1, len(V)):
        v = V[i]
        if np.dot(v, Vc[-1]) < 0:
            v = -v
        Vc.append(v)
    Vc = np.array(Vc)

    # --- 5. Signed incremental angles ---
    def signed_angle(u, v):
        return np.arctan2(
            u[0]*v[1] - u[1]*v[0],
            u[0]*v[0] + u[1]*v[1]
        )

    dtheta = np.array([
        signed_angle(Vc[i], Vc[(i+1) % len(Vc)])
        for i in range(len(Vc))
    ])

    # --- 6. Rotation-density–weighted inward normal ---
    direction = np.sum(np.abs(dtheta)[:, None] * normals, axis=0)

    norm = np.linalg.norm(direction)
    if norm == 0:
        return np.zeros(2)

    return direction / norm


# Setup struct
class mat(Structure):
    _fields_ = [('m11', c_double),
                ('m12', c_double),
               ('m21', c_double),
               ('m22', c_double)]

class vis_params(Structure):
    _fields_ = [('M', c_double),
                ('omega1', c_double),
                ('omega2', c_double),
                ('t', c_double), 
                ('C2', c_double),
                ('w2', c_double),
                ('ell', c_int),
                ('A_re', c_double*10146),
                ('A_im', c_double*10146),
                ('B_re', c_double*10146),
                ('B_im', c_double*10146),
                ('l', c_double*10146),
                ('m', c_double*10146),
                ('coef_length', c_int)]
    
class vect(Structure):
    _fields_ = [('x', c_double*10000),
                ('y', c_double*10000),
                ('z', c_double*10000), 
                ('m', c_double*10000),
                ('its', c_int)]
    
class rk_params(Structure):
    _fields_ = [('M', c_double),
                ('omega', c_double),
                ('t', c_double), 
                ('C', c_double),
                ('w', c_double)]

def to_c_array(row, size=10146):
    data = row.tolist()
    if len(data) < size:
        data += [0.0] * (size - len(data))
    return (c_double * size)(*data[:size])

def make_vis_param(lMax, name):
    df = pd.read_csv(name, header=None, index_col=0)

    # ------------------ Parameters ------------------
    A_re = np.array(df.iloc[0].tolist(), dtype='float64')
    A_im = np.array(df.iloc[1].tolist(), dtype='float64')
    B_re = np.array(df.iloc[2].tolist(), dtype='float64')
    B_im = np.array(df.iloc[3].tolist(), dtype='float64')
    M = np.array(df.iloc[4].tolist(), dtype='int')
    L = np.array(df.iloc[5].tolist(), dtype='int')

    coef_length = len(A_re)

    A = {}
    B = {}
    for i, ml in enumerate(zip(M, L)):
        m, l = ml
        ml = (int(m), int(l))
        A[ml] = A_re[i] + (1.j * A_im[i])
        B[ml] = B_re[i] + (1.j * B_im[i])

    print(f"Size: {len(A_im)}. Max: 10146.")

    model_param = vis_params(
        M=1.,
        omega1=1,
        omega2=1,
        t=0.,
        C2=.1,
        w2=np.pi/4,
        ell=lMax,
        A_re=to_c_array(A_re),
        A_im=to_c_array(A_im),
        B_re=to_c_array(B_re),
        B_im=to_c_array(B_im),
        m=to_c_array(M),
        l=to_c_array(L),
        coef_length=coef_length
    )

    return model_param, A, B

def force_z_calc(lMax, lMin, A, B, Omega= 1, r=1):
    """
    Calculate the z-component of force: F_z
    
    F_z^{lm} = a [2(l-m+1)(l+m+1)]^{1/2} (8r²/Ω²) (Ā^{lm} A^{l+1,m} + B̄^{lm} B^{l+1,m})
             + (-i m / (8π l(l+1))) (8r²/Ω²) Ā^{lm} B^{lm}
    """
    tot = 0
    prefactor = 8.0 * r * r / (Omega * Omega)
    
    for l in range(lMin, lMax+1):
        for m in range(-l, l+1):
            Alm = R(A, l, m)
            Alp1m = R(A, l+1, m)
            Blm = R(B, l, m)
            Blp1m = R(B, l+1, m)

            # Alpha term: l ↔ l+1, m stays same
            a_coef = 1.0 / (32.0 * np.pi * (l + 1)) * np.sqrt((2.0 * (l - 1) * (l + 3)) / ((2.0 * l + 1) * (2.0 * l + 3)))
            alpha = a_coef * np.sqrt(2.0 * (l - m + 1) * (l + m + 1)) * prefactor

            aa_term = alpha * (np.conj(Alm) * Alp1m + np.conj(Blm) * Blp1m)

            # Beta term: A ↔ B at same (l,m)
            beta = (-1.j * m / (8.0 * np.pi * l * (l+1))) * prefactor
            ab_term = beta * np.conj(Alm) * Blm

            tot += aa_term + ab_term

    return np.real(tot)

def force_x_calc(lMax, lMin, A, B, Omega= 1, r=1):
    """
    Calculate the x-component of force: F_x
    
    Based on equation with multiple coupling terms:
    - Alpha terms: coupling (l,m) ↔ (l+1, m±1)
    - Beta terms: coupling (l,m,A) ↔ (l, m±1, B)
    """
    tot = 0
    prefactor = 8.0 * r * r / (Omega * Omega)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    
    for l in range(lMin, lMax+1):
        a_coef = 1.0 / (32.0 * np.pi * (l + 1)) * np.sqrt((2.0 * (l - 1) * (l + 3)) / ((2.0 * l + 1) * (2.0 * l + 3)))
        b_pf = prefactor / (4.0 * np.pi * l * (l + 1))
        
        for m in range(-l, l+1):
            Alm = R(A, l, m)
            Blm = R(B, l, m)
            Alm_bar = np.conj(Alm)
            Blm_bar = np.conj(Blm)

            # ── Alpha terms: l ↔ l+1, m shifts by ±1 ──────────────────────
            if l + 1 <= lMax:
                # T11: coupling (l,m) ↔ (l+1, m-1)
                sqrt_factor_m = np.sqrt((l - m + 1) * (l - m + 2))
                K_m = a_coef * prefactor * inv_sqrt2 * sqrt_factor_m
                
                Alp1mm1 = R(A, l+1, m-1)
                Blp1mm1 = R(B, l+1, m-1)
                alpha_m = K_m * (Alm_bar * Alp1mm1 + Blm_bar * Blp1mm1)
                
                # T13: coupling (l,m) ↔ (l+1, m+1)
                sqrt_factor_p = np.sqrt((l + m + 1) * (l + m + 2))
                K_p = -a_coef * prefactor * inv_sqrt2 * sqrt_factor_p
                
                Alp1mp1 = R(A, l+1, m+1)
                Blp1mp1 = R(B, l+1, m+1)
                alpha_p = K_p * (Alm_bar * Alp1mp1 + Blm_bar * Blp1mp1)
                
                tot += alpha_m + alpha_p

            # ── Beta terms: A_{l,m} ↔ B_{l, m±1} ──────────────────────────
            # T21: coupling (l,m,A) ↔ (l, m-1, B)
            if m > -l:
                mag_m = b_pf * np.sqrt(0.5 * (l + m) * (l - m + 1))
                K_beta_m = -1.j * mag_m
                Blmm1 = R(B, l, m - 1)
                beta_m = K_beta_m * Alm_bar * Blmm1
                tot += beta_m

            # T23: coupling (l,m,A) ↔ (l, m+1, B)
            if m < l:
                mag_p = b_pf * np.sqrt(0.5 * (l - m) * (l + m + 1))
                K_beta_p = 1.j * mag_p
                Blmp1 = R(B, l, m + 1)
                beta_p = K_beta_p * Alm_bar * Blmp1
                tot += beta_p

    return np.real(tot)

def force_y_calc(lMax, lMin, A, B, Omega= 1, r=1):
    """
    Calculate the y-component of force: F_y
    
    Based on equation with multiple coupling terms (similar to F_x but with sign changes):
    - Alpha terms: coupling (l,m) ↔ (l+1, m±1) with -i factor
    - Beta terms: coupling (l,m,A) ↔ (l, m±1, B)
    """
    tot = 0
    prefactor = 8.0 * r * r / (Omega * Omega)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    
    for l in range(lMin, lMax+1):
        a_coef = 1.0 / (32.0 * np.pi * (l + 1)) * np.sqrt((2.0 * (l - 1) * (l + 3)) / ((2.0 * l + 1) * (2.0 * l + 3)))
        b_pf = prefactor / (4.0 * np.pi * l * (l + 1))
        
        for m in range(-l, l+1):
            Alm = R(A, l, m)
            Blm = R(B, l, m)
            Alm_bar = np.conj(Alm)
            Blm_bar = np.conj(Blm)

            # ── Alpha terms: l ↔ l+1, m shifts by ±1 (with -i factor) ────
            if l + 1 <= lMax:
                # T11y: coupling (l,m) ↔ (l+1, m-1) with -i factor
                sqrt_factor_m = np.sqrt((l - m + 1) * (l - m + 2))
                K_m_mag = a_coef * prefactor * inv_sqrt2 * sqrt_factor_m
                K_m = -1.j * K_m_mag
                
                Alp1mm1 = R(A, l+1, m-1)
                Blp1mm1 = R(B, l+1, m-1)
                alpha_m = K_m * (Alm_bar * Alp1mm1 + Blm_bar * Blp1mm1)
                
                # T13y: coupling (l,m) ↔ (l+1, m+1) with +i factor
                sqrt_factor_p = np.sqrt((l + m + 1) * (l + m + 2))
                K_p_mag = a_coef * prefactor * inv_sqrt2 * sqrt_factor_p
                K_p = 1.j * K_p_mag
                
                Alp1mp1 = R(A, l+1, m+1)
                Blp1mp1 = R(B, l+1, m+1)
                alpha_p = K_p * (Alm_bar * Alp1mp1 + Blm_bar * Blp1mp1)
                
                tot += alpha_m + alpha_p

            # ── Beta terms: A_{l,m} ↔ B_{l, m±1} ──────────────────────────
            # T21y: coupling (l,m,A) ↔ (l, m-1, B)
            if m > -l:
                mag_m = b_pf * np.sqrt(0.5 * (l + m) * (l - m + 1))
                K_beta_m = -mag_m
                Blmm1 = R(B, l, m - 1)
                beta_m = K_beta_m * Alm_bar * Blmm1
                tot += beta_m

            # T23y: coupling (l,m,A) ↔ (l, m+1, B)
            if m < l:
                mag_p = b_pf * np.sqrt(0.5 * (l - m) * (l + m + 1))
                K_beta_p = mag_p
                Blmp1 = R(B, l, m + 1)
                beta_p = K_beta_p * Alm_bar * Blmp1
                tot += beta_p

    return np.real(tot)

def force_calc(lMax, lMin, A, B, Omega= 1, r=1):
    Fx = force_x_calc(lMax, lMin, A, B, Omega, r)
    Fy = force_y_calc(lMax, lMin, A, B, Omega, r)
    Fz = force_z_calc(lMax, lMin, A, B, Omega, r)

    F = np.sqrt(Fx**2 + Fy**2 + Fz**2)
    return F

def R(T, l, m):
    try:
        val = T[(m, l)]
        if not np.isnan(val):
            return val
        else:
            return 0
    except:
        return 0

def load_coefs(name, offset=1, norm = False):
    df = pd.read_csv(name, header=None, index_col=0)

    A_re = np.array(df.iloc[0].tolist(), dtype='complex128')
    A_im = np.array(df.iloc[1].tolist(), dtype='complex128')
    B_re = np.array(df.iloc[2].tolist(), dtype='complex128')
    B_im = np.array(df.iloc[3].tolist(), dtype='complex128')
    M = np.array(df.iloc[4].tolist(), dtype='int')
    L = np.array(df.iloc[5].tolist(), dtype='int')

    s1 = np.sum(np.sqrt(A_re**2 + A_im**2) + np.sqrt(B_re**2 + B_im**2))
    s2 = np.sqrt(np.sum(np.abs(A_re + 1.j*A_im)**2 + np.abs(B_re + 1.j* B_im)**2))

    s = s2
    if norm:
        A_re /= s
        A_im /= s
        B_re /= s
        B_im /= s

    A = {}
    B = {}

    for i, ml in enumerate(zip(M, L)):
        m, l = ml
        ml = (int(m), int(l))
        A[ml] = (A_re[i] + (1.j * A_im[i])) * offset
        B[ml] = (B_re[i] + (1.j * B_im[i])) * offset

    return A, B

def force_xyz_calc(lMax, A, B, Omega=1, r=1, lMin=2):
    """
    Calculate all three force components F_x, F_y, F_z simultaneously.
    
    Returns:
        tuple: (F_x, F_y, F_z) as real numbers
    """
    Fx = force_x_calc(lMax, A, B, Omega, r, lMin)
    Fy = force_y_calc(lMax, A, B, Omega, r, lMin)
    Fz = force_z_calc(lMax, A, B, Omega, r, lMin)
    return Fx, Fy, Fz

def force_magnitude(lMax, A, B, Omega=1, r=1, lMin=2):
    """
    Calculate the magnitude of the total force: |F| = sqrt(F_x² + F_y² + F_z²)
    """
    Fx, Fy, Fz = force_xyz_calc(lMax, A, B, Omega, r, lMin)
    return np.sqrt(Fx**2 + Fy**2 + Fz**2)

def full_force_calc(lMax, A, B, omega, r=1):
    """
    Legacy function: Calculate all three force components as a vector.
    
    Returns:
        ndarray: [F_x, F_y, F_z]
    """
    Fx = force_x_calc(lMax, A, B, omega, r, lMin=2)
    Fy = force_y_calc(lMax, A, B, omega, r, lMin=2)
    Fz = force_z_calc(lMax, A, B, omega, r, lMin=2)
    return np.array([Fx, Fy, Fz])

def spherical_to_xyz(r, theta, phi):
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)
    return x, y, z

def spherical_fibonacci_points(N, R):
    """
    Generate N points evenly distributed on a unit sphere.
    Returns theta (polar angle) and phi (azimuthal angle) arrays.
    """
    # Golden ratio
    gr = (1 + np.sqrt(5)) / 2

    # Index array
    i = np.arange(N)

    # z goes from 1 to -1
    z = 1 - (2*i + 1)/N
    theta = np.arccos(z)  # polar angle

    phi = 2 * np.pi * i / gr  # azimuthal angle
    phi = np.mod(phi, 2*np.pi)  # ensure 0 <= phi < 2pi

    r = np.zeros(N) + R

    return r, theta, phi

def make_seeds(N, R, rand=False, cart = False):
    if rand:
        theta = np.random.rand(N) * 2 * np.pi
        u = np.random.rand(N)
    else:
        theta = np.linspace(0,2 * np.pi, N)
        u =  np.linspace(0,1,N)

    phi = np.acos(1 - (2*u))
    r = np.zeros(N) + R
    #theta = np.full(N,.5)

    if cart:
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)

        return x, y, z

    return r, theta, phi

def fit_circle(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x**2 + y**2

    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = c[0], c[1]
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r

def sphere_plot(vals, ax, n_theta=128, n_phi=256, R=1):

    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    Phi, Theta = np.meshgrid(phi, theta)

    X = R * np.sin(Theta) * np.cos(Phi)
    Y = R * np.sin(Theta) * np.sin(Phi)
    Z = R * np.cos(Theta)

    # Normalize values to [0,1]
    normed = vals
    vmin, vmax = vals.min(), vals.max()
    normed = (vals - vmin) / (vmax - vmin + 1e-12)
    colors_temp = plt.cm.viridis(normed)

    # Plot sphere
    surf = ax.plot_surface(
        X, Y, Z,
        facecolors=colors_temp,
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False
    )

