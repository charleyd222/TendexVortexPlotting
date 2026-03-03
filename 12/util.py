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
                ('A_re', c_double*10142),
                ('A_im', c_double*10142),
                ('B_re', c_double*10142),
                ('B_im', c_double*10142),
                ('l', c_double*10142),
                ('m', c_double*10142),
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

def to_c_array(row, size=10142):
    data = row.tolist()
    if len(data) < size:
        data += [0.0] * (size - len(data))
    return (c_double * size)(*data[:size])


def make_vis_param(lMax, name):
    df = pd.read_csv(name, header=None, index_col=0)

    # ------------------ Parameters ------------------
    A_re = np.array(df.iloc[0].tolist())
    A_im = np.array(df.iloc[1].tolist())
    B_re = np.array(df.iloc[2].tolist())
    B_im = np.array(df.iloc[3].tolist())
    M = np.array(df.iloc[4].tolist(), dtype='int')
    L = np.array(df.iloc[5].tolist(), dtype='int')

    coef_length = len(A_re) - 1


    A = {}
    B = {}
    for i, ml in enumerate(zip(M, L)):
        m, l = ml
        ml = (int(m), int(l))
        A[ml] = A_re[i] + (1.j * A_im[i])
        B[ml] = B_re[i] + (1.j * B_im[i])

    print(f"Size: {len(A_im)}. Max: 10142.")

    print(A_im, B_re)

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

def force_calc(lMax, omega, A, B):
    tot = 0
    for l in range(2,lMax+1):
        for mI in range(1,2 * l + 1):
            m = mI - l
            if abs(m) == 2:
                Alm = R(A,l,m)
                Alp1m = R(A,l+1,m)
                Blm = R(B,l,m)
                Blp1m = R(B,l+1,m)
                I_S_fact = (4 / (32 * np.pi * (l + 1))) * np.sqrt((2 * (l - 1) * (l + 3)) / ((2 * l + 1) * (2 * l + 3))) * np.sqrt(2 * (l - m + 1) * (l + m + 1))
                C_fact = (-1.j /(8 * np.pi * l * (l+1)) )
                I = np.conj(Alm) * Alp1m
                S = np.conj(Blm) * Blp1m
                C = m * np.conj(Alm) * Blm
                if np.isnan((64 / omega**2) * (I_S_fact * (I + S) + C_fact * C)):
                    print((64 / omega**2) * (I_S_fact * (I + S) + C_fact * C))
                tot += (64 / omega**2) * (I_S_fact * (I + S) + C_fact * C)


    return np.real(tot)

def R(T, l, m):
    try:
        val = T[(m, l)]
        if not np.isnan(val):
            return val
        else:
            return 0
    except:
        return 0

def full_force_calc(lMax, omega, A, B, r=1):

    Xi_m1 = (1 / np.sqrt(2)) * np.array([1, -1.j, 0])
    Xi_0 = np.array([0, 0, 1])
    Xi_p1 = (-1 / np.sqrt(2)) * np.array([1, 1.j, 0])

    tot = np.zeros(3, dtype='complex128')
    for l in range(2,lMax+1):
        for mI in range(1, 2 * l + 1):
            a = 1/(32 * np.pi * (l+1)) * ((2 * (l-1) * (l+3)) / ((2 * l + 1) * (2 * l + 3)))**(.5)
            m = mI - l

            Alm_bar = np.conj(R(A, l, m))
            Alp1mm1 = Alm_bar * R(A, l+1, m-1)
            Alp1m = Alm_bar * R(A, l+1, m)
            Alp1mp1 = Alm_bar * R(A, l+1, m+1)

            Blm = R(B, l, m)
            Blm_bar = np.conj(Blm)
            Blp1mm1 = Blm_bar * R(B, l+1, m-1)
            Blp1m = Blm_bar * R(B, l+1, m)
            Blp1mp1 = Blm_bar * R(B, l+1, m+1)
            Blmm1 = R(B, l, m-1)
            Blmp1 = R(B, l, m+1)

            T11 = np.sqrt((l-m+1) * (l-m+2)) * (Alp1mm1 + Blp1mm1) * Xi_m1
            T12 = np.sqrt(2 * (l-m+1) * (l+m+1)) * (Alp1m + Blp1m) * Xi_0
            T13 = np.sqrt((l+m+1) * (l+m+2)) * (Alp1mp1 + Blp1mp1) * Xi_p1
            T1 = a * (8 * r / omega)**2 * (T11 + T12 + T13)

            T21 = np.sqrt(.5 * (l+m) * (l-m+1)) * Blmm1 * Xi_m1
            T22 = m * Blm * Xi_0
            T23 = -1 * np.sqrt(.5 * (l-m) * (l+m+1)) * Blmp1 * Xi_p1
            T2 = (-1j / (8 * np.pi * (l + 1))) * (8 * r / omega)**2 * Alm_bar * (T21 + T22 + T23)

            tot += T1 + T2
            
    return tot.real

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

