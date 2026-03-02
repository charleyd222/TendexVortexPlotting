"""
test_swsh_plot.py  —  SWSH accuracy benchmark
==============================================
Compares the C++ spin_weighted_sh implementation against moble's
Python `spherical` package across a grid of (ell, m) values,
then plots heatmaps of the absolute error for s=+2 and s=-2.

Usage
-----
    # Build the shared library first, e.g.:
    #   g++ -O2 -shared -fPIC -o cppScripts/swsh_test.so swsh_test.cpp
    python test_swsh_plot.py

Outputs
-------
    swsh_error_heatmap.png   — four-panel heatmap figure
    swsh_error_stats.txt     — per-(ell,m) statistics

Requirements
------------
    numpy, matplotlib, quaternionic, spherical, ctypes, tqdm
"""

import sys
import ctypes
import pathlib

import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import quaternionic
import spherical

# ── Configuration ────────────────────────────────────────────────────────────

ELL_MAX   = 100          # maximum ell to test  (keep ≤ 50 for speed)
ELL_MIN   = 2           # must match Wigner(ELL_MAX, ELL_MIN)
N_ANGLE   = 2           # angles sampled per axis (N_ANGLE² points per (ell,m))
LIB_PATH  = pathlib.Path("./cppScripts/swsh_test")  # adjust if needed

# ── Load shared library ───────────────────────────────────────────────────────

class _Vect(ctypes.Structure):
    _fields_ = [
        ("Yp2_re", ctypes.c_double),
        ("Yp2_im", ctypes.c_double),
        ("Ym2_re", ctypes.c_double),
        ("Ym2_im", ctypes.c_double),
    ]

def load_lib(path: pathlib.Path):
    if not path.exists():
        sys.exit(f"[ERROR] Shared library not found: {path}\n"
                 f"Build it with:\n"
                 f"  g++ -O2 -shared -fPIC -std=c++17 "
                 f"-o {path} swsh_test.cpp")
    lib = ctypes.CDLL(str(path))
    lib.swsh_test.argtypes = [
        ctypes.c_double,  # theta
        ctypes.c_double,  # phi
        ctypes.c_int,     # ell
        ctypes.c_int,     # m
    ]
    lib.swsh_test.restype = _Vect
    return lib

lib = load_lib(LIB_PATH)

# ── Set up moble/spherical reference ─────────────────────────────────────────

print(f"[INFO] Initialising spherical.Wigner({ELL_MAX}, {ELL_MIN}) …", flush=True)
wigner = spherical.Wigner(ELL_MAX, ELL_MIN)

# Build a fixed grid of (theta, phi) points
rng      = np.random.default_rng(42)
thetas   = rng.uniform(0.05, np.pi - 0.05, N_ANGLE)   # avoid poles
phis     = rng.uniform(0.0,  2 * np.pi,    N_ANGLE)

# Precompute all moble reference values at once (much faster than one-by-one)
print(f"[INFO] Computing moble reference values for {N_ANGLE}² = "
      f"{N_ANGLE**2} points …", flush=True)
# quaternionic expects shape (..., 2): [[theta, phi], ...]
angle_grid = np.array([[t, p] for t in thetas for p in phis])  # (N_ANGLE², 2)
R_all = quaternionic.array.from_spherical_coordinates(angle_grid)

Y_p2_ref = wigner.sYlm(+2, R_all)   # shape (N_ANGLE², n_modes)
Y_m2_ref = wigner.sYlm(-2, R_all)   # shape (N_ANGLE², n_modes)

# ── Sweep over (ell, m) ───────────────────────────────────────────────────────

ells = np.arange(ELL_MIN, ELL_MAX + 1)          # [2, 3, ..., ELL_MAX]
n_ell = len(ells)

# We'll store the maximum absolute error over all sample points for each (ell,m).
# Use a 2-D array indexed by [ell_idx, m+ELL_MAX] for convenience; fill with NaN
# for invalid (|m| > ell) combinations.
err_p2 = np.full((n_ell, 2 * ELL_MAX + 1), np.nan)
err_m2 = np.full((n_ell, 2 * ELL_MAX + 1), np.nan)

total_modes = sum(2 * ell + 1 for ell in ells)

print(f"[INFO] Sweeping {total_modes} (ell, m) combinations …", flush=True)

pbar = tqdm(
    total=total_modes,
    desc="  scanning (ell, m)",
    unit="mode",
    bar_format=(
        "  {l_bar}{bar}| {n_fmt}/{total_fmt} modes "
        "[{elapsed}<{remaining}, {rate_fmt}]"
    ),
    colour="yellow",
    dynamic_ncols=True,
)

for ell_idx, ell in enumerate(ells):
    for m in range(-ell, ell + 1):
        midx = wigner.Yindex(ell=int(ell), m=int(m))

        max_err_p2 = 0.0
        max_err_m2 = 0.0

        for pt_idx, (theta, phi) in enumerate(angle_grid):
            v = lib.swsh_test(float(theta), float(phi), int(ell), int(m))

            # C++ result
            cpp_p2 = complex(v.Yp2_re, v.Yp2_im)
            cpp_m2 = complex(v.Ym2_re, v.Ym2_im)

            # Python reference
            ref_p2 = complex(Y_p2_ref[pt_idx, midx])
            ref_m2 = complex(Y_m2_ref[pt_idx, midx])

            max_err_p2 = max(max_err_p2, abs(cpp_p2 - ref_p2))
            max_err_m2 = max(max_err_m2, abs(cpp_m2 - ref_m2))

        col = int(m) + ELL_MAX
        err_p2[ell_idx, col] = max_err_p2
        err_m2[ell_idx, col] = max_err_m2

        pbar.set_postfix(ell=int(ell), m=int(m),
                         err=f"{max(max_err_p2, max_err_m2):.2e}")
        pbar.update(1)

pbar.close()

# ── Save statistics ───────────────────────────────────────────────────────────

stats_path = pathlib.Path("swsh_error_stats.txt")
with open(stats_path, "w") as f:
    f.write(f"{'ell':>5}  {'m':>5}  {'max_err_s+2':>18}  {'max_err_s-2':>18}\n")
    f.write("-" * 55 + "\n")
    for ell_idx, ell in enumerate(ells):
        for m in range(-ell, ell + 1):
            col = int(m) + ELL_MAX
            ep = err_p2[ell_idx, col]
            em = err_m2[ell_idx, col]
            f.write(f"{ell:>5}  {m:>5}  {ep:>18.6e}  {em:>18.6e}\n")
print(f"[INFO] Stats written to {stats_path}")

# ── Build square heatmap arrays (ell vs m) ───────────────────────────────────
# Trim the m axis to just [-ELL_MAX, ELL_MAX] and keep NaN where invalid.
# x-axis: m  (columns), y-axis: ell (rows, low ell at top → flip for display)

m_axis   = np.arange(-ELL_MAX, ELL_MAX + 1)   # length 2*ELL_MAX+1
ell_axis = ells                                 # length n_ell

# Clamp near-zero values to a floor so log scale works cleanly
FLOOR = 1e-16
ep_plot = np.where(np.isfinite(err_p2), np.maximum(err_p2, FLOOR), np.nan)
em_plot = np.where(np.isfinite(err_m2), np.maximum(err_m2, FLOOR), np.nan)

# ── Plotting ─────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.family":      "monospace",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

# Shared colour scale across all panels
vmin = FLOOR
vmax = max(np.nanmax(ep_plot), np.nanmax(em_plot))
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
cmap = plt.get_cmap("inferno")

fig, axes = plt.subplots(
    2, 2,
    figsize=(16, 11),
    gridspec_kw={"hspace": 0.38, "wspace": 0.12},
)

LABEL_COLOR  = "#e8e0d0"
TITLE_COLOR  = "#f5c842"
GRID_COLOR   = "#2a2a2a"
ACCENT_COLOR = "#ff6b35"

def style_ax(ax):
    ax.tick_params(colors=LABEL_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)

# ── Panel helper ─────────────────────────────────────────────────────────────

def make_panel(ax, data, title, show_ylabel=True):
    """Draw one heatmap panel."""
    style_ax(ax)

    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[m_axis[0] - 0.5, m_axis[-1] + 0.5,
                ell_axis[0]  - 0.5, ell_axis[-1] + 0.5],
        norm=norm,
        cmap=cmap,
        interpolation="nearest",
    )

    # Diagonal guides: m = ±ell boundaries
    ax.plot([ell_axis[0], ell_axis[-1]],   [ell_axis[0], ell_axis[-1]],
            color="#ffffff", lw=0.5, ls="--", alpha=0.4, label="|m|=ℓ")
    ax.plot([-ell_axis[0], -ell_axis[-1]], [ell_axis[0], ell_axis[-1]],
            color="#ffffff", lw=0.5, ls="--", alpha=0.4)

    ax.set_xlim(-ELL_MAX - 1, ELL_MAX + 1)
    ax.set_ylim(ell_axis[0] - 1, ell_axis[-1] + 1)

    ax.set_xlabel("m", color=LABEL_COLOR, fontsize=9, labelpad=4)
    if show_ylabel:
        ax.set_ylabel("ℓ", color=LABEL_COLOR, fontsize=9, labelpad=4)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    ax.set_title(title, color=TITLE_COLOR, fontsize=11,
                 fontweight="bold", pad=8)
    return im

# ── Panel 1 & 2: raw heatmaps ────────────────────────────────────────────────

im = make_panel(axes[0, 0], ep_plot, "s = +2  │  max |error| over angles")
make_panel(axes[0, 1], em_plot, "s = −2  │  max |error| over angles",
           show_ylabel=False)

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.54, 0.013, 0.38])
cb = fig.colorbar(im, cax=cbar_ax, norm=norm)
cb.set_label("Absolute error", color=LABEL_COLOR, fontsize=8)
cb.ax.yaxis.set_tick_params(color=LABEL_COLOR, labelsize=7)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=LABEL_COLOR)
cb.outline.set_edgecolor(GRID_COLOR)

# ── Panel 3: error vs ell (max over all m and angles) ────────────────────────

ax3 = axes[1, 0]
style_ax(ax3)

max_by_ell_p2 = np.nanmax(ep_plot, axis=1)   # max over m
max_by_ell_m2 = np.nanmax(em_plot, axis=1)

ax3.semilogy(ells, max_by_ell_p2, color="#f5c842", lw=1.8,
             label="s = +2", marker="o", markersize=2.5)
ax3.semilogy(ells, max_by_ell_m2, color=ACCENT_COLOR, lw=1.8,
             ls="--", label="s = −2", marker="s", markersize=2.5)

ax3.axhline(1e-10, color="#888", lw=0.7, ls=":", alpha=0.7)
ax3.text(ELL_MAX * 0.55, 1.5e-10, "threshold 1e-10",
         color="#aaa", fontsize=7)

ax3.set_xlabel("ℓ",            color=LABEL_COLOR, fontsize=9)
ax3.set_ylabel("max |error|",  color=LABEL_COLOR, fontsize=9)
ax3.set_title("Error growth vs ℓ  (worst m)",
              color=TITLE_COLOR, fontsize=11, fontweight="bold", pad=8)
ax3.legend(edgecolor=GRID_COLOR,
           labelcolor=LABEL_COLOR, fontsize=8)
ax3.tick_params(colors=LABEL_COLOR, labelsize=8)
ax3.set_xlim(ells[0], ells[-1])

# ── Panel 4: error vs |m|/ell  (normalised magnetic index) ───────────────────

ax4 = axes[1, 1]
style_ax(ax4)

# Scatter: x = |m|/ell,  y = error,  colour = ell
scatter_x_p2, scatter_y_p2, scatter_c = [], [], []
for ell_idx, ell in enumerate(ells):
    for m in range(-ell, ell + 1):
        col = int(m) + ELL_MAX
        e = ep_plot[ell_idx, col]
        if np.isfinite(e):
            scatter_x_p2.append(abs(m) / ell)
            scatter_y_p2.append(e)
            scatter_c.append(ell)

scatter_x_p2 = np.array(scatter_x_p2)
scatter_y_p2 = np.array(scatter_y_p2)
scatter_c    = np.array(scatter_c, dtype=float)

sc = ax4.scatter(
    scatter_x_p2, scatter_y_p2,
    c=scatter_c, cmap="plasma",
    norm=mcolors.Normalize(vmin=ELL_MIN, vmax=ELL_MAX),
    s=3, alpha=0.6, linewidths=0,
)
ax4.set_yscale("log")
ax4.set_xlabel(r"|m| / $\ell$   (0 = axial,  1 = equatorial)",
               color=LABEL_COLOR, fontsize=8)
ax4.set_ylabel("max |error|", color=LABEL_COLOR, fontsize=9)
ax4.set_title(r"Error vs normalised |m|  (s = +2, coloured by $\ell$)",
              color=TITLE_COLOR, fontsize=11, fontweight="bold", pad=8)
ax4.tick_params(colors=LABEL_COLOR, labelsize=8)

cbar2_ax = fig.add_axes([0.92, 0.07, 0.013, 0.38])
cb2 = fig.colorbar(sc, cax=cbar2_ax)
cb2.set_label(r"$\ell$", color=LABEL_COLOR, fontsize=8)
cb2.ax.yaxis.set_tick_params(color=LABEL_COLOR, labelsize=7)
plt.setp(cb2.ax.yaxis.get_ticklabels(), color=LABEL_COLOR)
cb2.outline.set_edgecolor(GRID_COLOR)

# ── Super-title ───────────────────────────────────────────────────────────────

fig.suptitle(
    f"SWSH C++ vs moble/spherical  │  ℓ ∈ [{ELL_MIN}, {ELL_MAX}]  │  "
    f"{N_ANGLE}² angle samples",
    color=LABEL_COLOR, fontsize=13, fontweight="bold", y=0.98,
)

out_path = pathlib.Path("swsh_error_heatmap.png")
fig.savefig(out_path, dpi=160, bbox_inches="tight")
print(f"[INFO] Saved → {out_path}")
plt.show()