import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import matplotlib.colors as mcolors

def sech(x):
    return 1/np.cosh(x)
def coth(x):
    return 1/np.tanh(x)

def f(x,y,z,R,S): # Negative E Val
    r = np.sqrt((z*z) + (y*y) + (x*x))

    sechPlus2 = sech(S*(r + R)) * sech(S*(r + R))
    sechMin2 = sech(S*(r - R)) * sech(S*(r - R))
    sechMinPlus = sechMin2 - sechPlus2
    sechMinPlusTanh = sechMin2*np.tanh(S*(r-R)) - sechPlus2*np.tanh(S*(r+R))

    DzDyB = -.5 * coth(R*S) * S * y * z * (sechMinPlus / (r**3) + (2 * S * sechMinPlusTanh / r**2))
    DyDyB = -.5 * coth(R*S) * S * (y * y * (sechMinPlus / (r**3) + (2 * S * sechMinPlusTanh / r**2)) + (S * -1 * sechMinPlus / r))
    DzDzB = -.5 * coth(R*S) * S * (z * z * (sechMinPlus / (r**3) + (2 * S * sechMinPlusTanh / r**2)) + (S * -1 * sechMinPlus / r))

    return .5 * np.sqrt((DzDyB*DzDyB) - DyDyB*DzDzB)

N=200
# Define the grid
x = np.linspace(-5, 5, N)
y = np.linspace(-5, 5, N)
z = np.linspace(-5, 5, N)
y_grid, z_grid = np.meshgrid(y, z)
x_fixed = 0  # Initial x value
R_fixed = 5  # Initial R value
S_fixed = 5  # Initial S value

norm = mcolors.Normalize()

# Create the 2D heatmap with sliders
fig, ax2 = plt.subplots()

heatmap = ax2.contourf(y_grid, z_grid, f(x_fixed, y_grid, z_grid, R_fixed, S_fixed), levels=100, cmap='inferno', norm=norm)
cbar = fig.colorbar(heatmap, ax=ax2, label='Eigen Value', norm = norm)
ax2.set_xlabel('Y-axis')
ax2.set_ylabel('Z-axis')
ax2.set_title(f'Heatmap at x={x_fixed}, R={R_fixed}, S={S_fixed}')

# Sliders to adjust x, R, and S levels
ax_slider_x = plt.axes([0.25, 0.02, 0.5, 0.02])
slider_x = Slider(ax_slider_x, 'X Level', -50, 50, valinit=x_fixed)

ax_slider_R = plt.axes([0.25, 0.06, 0.5, 0.02])
slider_R = Slider(ax_slider_R, 'R Level', 1, 10, valinit=R_fixed)

ax_slider_S = plt.axes([0.25, 0.10, 0.5, 0.02])
slider_S = Slider(ax_slider_S, 'S Level', 1, 10, valinit=S_fixed)

def update(val):
    x_fixed = slider_x.val
    R_fixed = slider_R.val
    S_fixed = slider_S.val
    ax2.clear()
    new_heatmap = ax2.contourf(y_grid, z_grid, f(x_fixed, y_grid, z_grid, R_fixed, S_fixed), levels=100, cmap='inferno',norm=norm)
    ax2.set_xlabel('Y-axis')
    ax2.set_ylabel('Z-axis')
    ax2.set_title(f'Heatmap at x={x_fixed}, R={R_fixed}, S={S_fixed}')
    global heatmap
    heatmap = new_heatmap  # Update heatmap reference
    fig.canvas.draw_idle()

slider_x.on_changed(update)
slider_R.on_changed(update)
slider_S.on_changed(update)

plt.show()
