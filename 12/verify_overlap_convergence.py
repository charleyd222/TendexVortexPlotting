import argparse
import csv
import re
from dataclasses import dataclass
from math import pi
from pathlib import Path

import numpy as np
import quaternionic
import spherical
from scipy.special import roots_legendre
import matplotlib.pyplot as plt


MM = np.outer(np.array([0.0, 1.0, 1.0j]), np.array([0.0, 1.0, 1.0j]))
MBarMBar = np.outer(np.array([0.0, 1.0, -1.0j]), np.array([0.0, 1.0, -1.0j]))


@dataclass
class OverlapCoefficients:
    lmax: int
    coeffs: dict


def exact_e0(theta: float, phi: float, lam: float, dtheta: float) -> np.ndarray:
    ct = np.cos(theta)
    c2p = np.cos(2.0 * phi)
    s2p = np.sin(2.0 * phi)

    t = np.zeros((3, 3), dtype=complex)
    t[1, 1] = c2p * (ct ** 2)
    t[1, 2] = -s2p * ct
    t[2, 1] = -s2p * ct
    t[2, 2] = c2p
    return t * lam * np.exp(-(theta ** 2) / (2.0 * dtheta ** 2))


def parse_float_row(row):
    values = []
    for item in row:
        item = item.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError:
            if len(values) == 0:
                continue
            raise
    return values


def load_overlap_csv(path: Path) -> OverlapCoefficients:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 6:
        raise ValueError(f"Expected at least 6 rows in {path}, found {len(rows)}")

    # Support both formats used in this repo:
    # 1) raw overlap.cpp output: numeric rows only
    # 2) overlap_integral.py output: labeled rows like A_re_V, A_im_V, ...
    if rows[0] and rows[0][0].strip() in {"A_re", "A_re_V"}:
        a_re = parse_float_row(rows[0][1:])
        a_im = parse_float_row(rows[1][1:])
        b_re = parse_float_row(rows[2][1:])
        b_im = parse_float_row(rows[3][1:])
        m_vals = [int(round(x)) for x in parse_float_row(rows[4][1:])]
        l_vals = [int(round(x)) for x in parse_float_row(rows[5][1:])]
    else:
        a_re = parse_float_row(rows[0])
        a_im = parse_float_row(rows[1])
        b_re = parse_float_row(rows[2])
        b_im = parse_float_row(rows[3])
        m_vals = [int(round(x)) for x in parse_float_row(rows[4])]
        l_vals = [int(round(x)) for x in parse_float_row(rows[5])]

    if not (len(a_re) == len(a_im) == len(b_re) == len(b_im) == len(m_vals) == len(l_vals)):
        raise ValueError(f"Row lengths do not match in {path}")

    coeffs = {}
    lmax = 0
    for ar, ai, br, bi, m_val, l_val in zip(a_re, a_im, b_re, b_im, m_vals, l_vals):
        coeffs[(l_val, m_val)] = (complex(ar, ai), complex(br, bi))
        lmax = max(lmax, l_val)

    return OverlapCoefficients(lmax=lmax, coeffs=coeffs)


def reconstruct_tensor(theta: float, phi: float, coeffs: OverlapCoefficients, wigner) -> np.ndarray:
    rotation = quaternionic.array.from_spherical_coordinates(np.array([theta]), np.array([phi]))
    y_plus = wigner.sYlm(2, rotation)
    y_minus = wigner.sYlm(-2, rotation)

    if y_plus.ndim == 2:
        y_plus = y_plus[0]
    if y_minus.ndim == 2:
        y_minus = y_minus[0]

    reconstructed = np.zeros((3, 3), dtype=complex)
    for (ell, m_val), (a_coeff, b_coeff) in coeffs.coeffs.items():
        mode_index = wigner.Yindex(ell, m_val)
        yp = y_plus[mode_index]
        ym = y_minus[mode_index]

        basis_a = MM * ym + MBarMBar * yp
        basis_b = -1.0j * (MM * ym - MBarMBar * yp)
        reconstructed += a_coeff * basis_a + b_coeff * basis_b

    return reconstructed


def relative_l2_error(coeffs: OverlapCoefficients, n_theta: int, n_phi: int, lam: float, dtheta: float):
    u_nodes, u_weights = roots_legendre(n_theta)
    theta_nodes = np.arccos(u_nodes)
    phi_nodes = 2.0 * pi * np.arange(n_phi) / n_phi
    phi_weight = 2.0 * pi / n_phi

    wigner = spherical.Wigner(coeffs.lmax, 2)

    err_sq = 0.0
    ref_sq = 0.0
    for theta, w_theta in zip(theta_nodes, u_weights):
        for phi in phi_nodes:
            e_exact = exact_e0(theta, phi, lam, dtheta)
            e_rec = reconstruct_tensor(theta, phi, coeffs, wigner)
            diff = e_rec - e_exact

            weight = w_theta * phi_weight
            err_sq += weight * np.sum(np.abs(diff) ** 2)
            ref_sq += weight * np.sum(np.abs(e_exact) ** 2)

    rel = np.sqrt(err_sq / ref_sq) if ref_sq > 0 else np.nan
    return rel, np.sqrt(err_sq), np.sqrt(ref_sq)


def find_csvs(folder: Path):
    return sorted(folder.glob("*.csv"))


def main():
    parser = argparse.ArgumentParser(description="Verify that overlap.cpp coefficients converge back to analytic E0.")
    parser.add_argument("--folder", default="overlap_csvs", help="Folder containing overlap.csv outputs")
    parser.add_argument("--n-theta", type=int, default=80, help="Quadrature points in theta")
    parser.add_argument("--n-phi", type=int, default=80, help="Uniform phi samples")
    parser.add_argument("--lam", type=float, default=1.0, help="E0 amplitude")
    parser.add_argument("--dtheta", type=float, default=0.01, help="Gaussian width in E0")
    parser.add_argument("--single", type=str, default=None, help="Verify one CSV file instead of scanning a folder")
    parser.add_argument("--plot", action="store_true", help="Plot relative error versus lMax")
    parser.add_argument("--save-plot", type=str, default=None, help="Save the plot to this path instead of showing it")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    if args.single is not None:
        single_path = Path(args.single)
        csv_paths = [single_path if single_path.is_absolute() else script_dir / single_path]
    else:
        folder_path = Path(args.folder)
        if not folder_path.is_absolute():
            folder_path = script_dir / folder_path
        csv_paths = find_csvs(folder_path)
        folder_desc = folder_path

    if not csv_paths:
        raise SystemExit(f"No overlap CSV files found in {folder_desc if args.single is None else Path(args.single)}.")

    print("lMax,relative_L2_error,abs_error,ref_norm")
    results = []
    for csv_path in csv_paths:
        coeffs = load_overlap_csv(csv_path)
        rel, abs_err, ref_norm = relative_l2_error(coeffs, args.n_theta, args.n_phi, args.lam, args.dtheta)
        results.append((coeffs.lmax, rel, abs_err, ref_norm, csv_path))
        print(f"{coeffs.lmax},{rel:.12e},{abs_err:.12e},{ref_norm:.12e}")

    if True:
        results.sort(key=lambda item: item[0])
        lmax_vals = [item[0] for item in results]
        rel_errors = [item[1] for item in results]
        abs_errors = [item[2] for item in results]

        fig_rel, ax_rel = plt.subplots(figsize=(7.5, 4.8))
        ax_rel.semilogy(lmax_vals, rel_errors, marker="o", lw=2, color="#1f77b4")
        ax_rel.set_xlabel("lMax")
        ax_rel.set_ylabel("Relative L2 error")
        ax_rel.set_title("Relative L2 error versus lMax")
        ax_rel.grid(True, which="both", ls=":", alpha=0.4)
        fig_rel.tight_layout()

        fig_abs, ax_abs = plt.subplots(figsize=(7.5, 4.8))
        ax_abs.semilogy(lmax_vals, abs_errors, marker="s", lw=2, color="#d62728")
        ax_abs.set_xlabel("lMax")
        ax_abs.set_ylabel("Absolute error")
        ax_abs.set_title("Absolute error versus lMax")
        ax_abs.grid(True, which="both", ls=":", alpha=0.4)
        fig_abs.tight_layout()

        if False:
            save_path = Path(args.save_plot)
            if save_path.suffix:
                rel_path = save_path.with_name(f"{save_path.stem}_relative{save_path.suffix}")
                abs_path = save_path.with_name(f"{save_path.stem}_absolute{save_path.suffix}")
            else:
                rel_path = save_path / "relative_l2_error.png"
                abs_path = save_path / "absolute_error.png"

            fig_rel.savefig(rel_path, dpi=200, bbox_inches="tight")
            fig_abs.savefig(abs_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot to {rel_path}")
            print(f"Saved plot to {abs_path}")
        else:
            plt.show()


if __name__ == "__main__":
    main()