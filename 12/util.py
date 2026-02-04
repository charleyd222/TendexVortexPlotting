from ctypes import *
import numpy as np
import matplotlib.pyplot as plt

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
                ('ell', c_int)]
    
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

