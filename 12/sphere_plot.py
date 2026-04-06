import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from ctypes import *
from datetime import datetime as dt
from util import vect, vis_params, make_vis_param, force_x_calc, force_y_calc, force_z_calc

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

lines = False
show_force_vectors_overlay = True
animate_time = False
save_animation = True
animation_file = 'sphere_time_animation.mp4'
num_frames = 20
time_span = 2 * np.pi
frame_interval_ms = 1000
#Runtime
start = dt.now()
fig = plt.figure(figsize=(7,7))
ax = []
ax += [fig.add_subplot(111, projection='3d')]
#ax += [fig.add_subplot(111, projection='3d')]

# Load cpp
val_return = CDLL("./cppScripts/rka_iter").val_return
val_return.argtypes = [c_double, c_double, c_double, c_int, vis_params, c_int]
val_return.restype = c_double

if lines:
    rka_iter = CDLL("./cppScripts/rka_iter").rka_iter
    rka_iter.argtypes = [c_double, c_double, c_double, c_int, c_int, c_double, c_double, c_double, c_double, vis_params]
    rka_iter.restype = vect

# Make data
gauss_dtheta = 0.001
lMax = 44
power = 1
#model_param, A, B = make_vis_param(lMax, f'cppScripts/csvs/power_1_omega_1/maximized_coefs_lMax_{lMax}_Power_1_power.csv')
model_param, A, B = make_vis_param(lMax, f'cppScripts/csvs/xyz_power_1_omega_1/maximized_coefs_xyz_lMax_{lMax}_Power_1_power.csv')
#model_param, A, B = make_vis_param(lMax, f'data_gauss_z_0p001_lMax_50.csv')

l = 50

# Resolution of the sphere grid
n_theta = 2*l
n_phi   = l
R   = 1.0

theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2*np.pi, n_phi)
Phi, Theta = np.meshgrid(phi, theta)

# Color the sphere
def compute_colors(t_value):
    model_param.t = float(t_value)
    vals = np.zeros_like(Theta)
    for i in range(Theta.shape[0]):
        for j in range(Theta.shape[1]):
            th = Theta[i, j]
            ph = Phi[i, j]
            vals[i, j] = val_return(R, th, ph, 1, model_param, 1)
    return vals

colors = compute_colors(0.0)
    
#ax.scatter(x,y,z, zorder=2)
X = R * np.sin(Theta) * np.cos(Phi)
Y = R * np.sin(Theta) * np.sin(Phi)
Z = R * np.cos(Theta)

# Normalize values to [0,1]
normed = colors
vmin, vmax = colors.min(), colors.max()
normed = (colors - vmin) / (vmax - vmin + 1e-12)

# Convert normalized values → RGBA  (this returns an (M,N,4) array)
colors_temp = plt.cm.viridis(normed)
#norm = mcolors.Normalize(vmin=0, vmax=np.max(colors))
#cmap = mat.colormaps['jet']
#colors_temp = cmap(norm(normed))

# Plot sphere
surf = ax[0].plot_surface(
    X, Y, Z,
    facecolors=colors_temp,
    rstride=1, cstride=1,
    linewidth=0,
    antialiased=False,
    shade=False
)

# Colorbar
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_clim(vmin, vmax)
fig.colorbar(mappable, shrink=0.6, label='value', ax=ax)
fig.suptitle(r'$\ell$' +f' between 2 and {lMax} linear combination. Spectral Methods')# + r'Maximizing $F =F_z$')#= \sqrt{F_x^2 + F_y^2 + F_z^2}$')

def draw_force_vectors(ax):
    Fx = force_x_calc(lMax, 2, A, B, Omega=1, r=R)
    Fy = force_y_calc(lMax, 2, A, B, Omega=1, r=R)
    Fz = force_z_calc(lMax, 2, A, B, Omega=1, r=R)
    Fmag = np.sqrt(Fx**2 + Fy**2 + Fz**2)

    max_comp = max(abs(Fx), abs(Fy), abs(Fz), Fmag, 1e-14)
    comp_scale = 2 / max_comp

    # Component arrows (x, y, z)
    ax.quiver(0, 0, 0, Fx * comp_scale, 0, 0, color='crimson', linewidth=2.2, arrow_length_ratio=0.12)
    ax.quiver(0, 0, 0, 0, Fy * comp_scale, 0, color='seagreen', linewidth=2.2, arrow_length_ratio=0.12)
    ax.quiver(0, 0, 0, 0, 0, Fz * comp_scale, color='royalblue', linewidth=2.2, arrow_length_ratio=0.12)

    # Total force direction arrow
    ax.quiver(0, 0, 0, Fx * comp_scale, Fy * comp_scale, Fz * comp_scale,
              color='black', linewidth=2.8, arrow_length_ratio=0.12)

    # Lightweight in-figure legend + values
    txt = (
        f"Fx={Fx:.4e}\n"
        f"Fy={Fy:.4e}\n"
        f"Fz={Fz:.4e}\n"
        f"|F|={Fmag:.4e}\n"
        f"colors: x=red, y=green, z=blue, total=black"
    )
    ax.text2D(0.02, 0.02, txt, transform=ax.transAxes, fontsize=9)

if show_force_vectors_overlay:
    draw_force_vectors(ax[0])

time_text = ax[0].text2D(0.02, 0.95, 't = 0.000', transform=ax[0].transAxes, fontsize=10)

def update(frame_idx):
    global surf
    t_value = (time_span * frame_idx) / max(1, (num_frames - 1))
    vals = compute_colors(t_value)

    vmin_t, vmax_t = vals.min(), vals.max()
    normed_t = (vals - vmin_t) / (vmax_t - vmin_t + 1e-12)
    colors_t = plt.cm.viridis(normed_t)

    surf.remove()
    surf = ax[0].plot_surface(
        X, Y, Z,
        facecolors=colors_t,
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False
    )

    mappable.set_clim(vmin_t, vmax_t)
    time_text.set_text(f't = {t_value:.3f}')
    return surf, time_text

if animate_time:
    anim = FuncAnimation(fig, update, frames=num_frames, interval=frame_interval_ms, blit=False)
    if save_animation:
        try:
            anim.save(animation_file, dpi=140)
            print(f'Saved animation to {animation_file}')
        except Exception as exc:
            print(f'Animation save failed: {exc}')

ax[0].set_box_aspect([1,1,1])
plt.show()
