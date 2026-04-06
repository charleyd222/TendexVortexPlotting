/*
 * multipole_opt_xy.cpp
 *
 * Maximises  |F_xy| = sqrt(Fx^2 + Fy^2)  subject to one of two constraints:
 *
 *   --constraint norm   ||c||² = l_max                          (default)
 *
 *   --constraint power  dE/dt = P  (uniform-power constraint)
 *       where  dE/dt = (r²)/(4π Ω²) · ||c||²
 *       so fixing dE/dt = P  ⟺  ||c||² = 4π Ω² P / r²
 *       The optional --power <P> flag sets the target power (default P=1).
 *
 * KEY INSIGHT (symmetry reduction):
 *   Rotating the field by φ around z maps c_{l,m} → e^{imφ} c_{l,m} and
 *   rotates (Fx,Fy) by φ.  Therefore max|F_xy| = max Fx = λ_max(M_x).
 *
 * KEY DIFFERENCE from Z-force:
 *   The x-force (T11/T13/T21/T23 terms) couples (l,m) to (l+1, m±1) and
 *   (l, m±1), so M_x is NOT block-diagonal in m.  We must build and
 *   diagonalise the full N×N matrix.
 *
 * Compile (macOS/Linux):
 *   clang++ -std=c++17 -O3 -march=native multipole_opt_xy.cpp -o multipole_opt_xy \
 *       -I /usr/local/include/eigen3
 *
 * Usage:
 *   ./multipole_opt_xy <l_max> [c] [P_target] [r] [Omega] [folder]
 *       c=0 → power constraint,  c=1 → norm constraint (default)
 *
 * Examples:
 *   ./multipole_opt_xy 10                              # norm constraint, l_max=10
 *   ./multipole_opt_xy 10 0 1.0                        # power constraint, P=1
 *   ./multipole_opt_xy 10 0 2.5 1.0 1.0 csvs          # power=2.5, r=1, Ω=1
 */

#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <sys/stat.h>

using cdouble   = std::complex<double>;
using MatrixXcd = Eigen::MatrixXcd;
using VectorXd  = Eigen::VectorXd;
using VectorXcd = Eigen::VectorXcd;
const double PI = std::acos(-1.0);

// ============================================================================
// Constraint type
// ============================================================================
enum class ConstraintType { Norm, Power };

// ---- Index map --------------------------------------------------------------
struct Key { int l, m, t;   // t=0 → A, t=1 → B
    bool operator<(const Key& o) const {
        if (l!=o.l) return l<o.l;
        if (m!=o.m) return m<o.m;
        return t<o.t; } };
using IndexMap = std::map<Key,int>;

IndexMap build_index_map(int l_max, int& N) {
    IndexMap idx; int i=0;
    for (int l=2; l<=l_max; ++l)
        for (int m=-l; m<=l; ++m) { idx[{l,m,0}]=i++; idx[{l,m,1}]=i++; }
    N=i; return idx;
}

// ---- Build full M_x matrix --------------------------------------------------
//
// For a Hermitian quadratic form c†Mc = Fx = Re(Σ K·conj(c_i)·c_j),
// each cross-term K·conj(c_i)·c_j with i≠j contributes M[i,j]=K/2,
// M[j,i]=conj(K)/2.  SelfAdjointEigenSolver reads only the lower triangle,
// but we write both halves for clarity and correctness checking.
//
MatrixXcd build_M_xy(int l_max, double r, double Omega,
                     const IndexMap& idx_map, int N) {
    MatrixXcd M = MatrixXcd::Zero(N, N);
    double pf = 8.0 * r * r / (Omega * Omega);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    for (int l = 2; l <= l_max; ++l) {
        // beta prefactor shared across m at this l
        double b_pf = pf / (4.0 * PI * l * (l + 1));

        for (int m = -l; m <= l; ++m) {
            int iA = idx_map.at({l, m, 0});
            int iB = idx_map.at({l, m, 1});

            // ── Alpha XY: l ↔ l+1, m shifts by ±1 ──────────────────────────
            if (l + 1 <= l_max) {
                double a = 1.0 / (32.0*PI*(l+1))
                         * std::sqrt((2.0*(l-1)*(l+3))
                                     / ((2.0*l+1)*(2.0*l+3)));

                // T11: coupling (l,m,A/B) ↔ (l+1, m-1, A/B)
                {
                    double K = a * pf * inv_sqrt2
                             * std::sqrt((double)(l - m + 1) * (l - m + 2));
                    int jA = idx_map.at({l+1, m-1, 0});
                    int jB = idx_map.at({l+1, m-1, 1});
                    M(iA, jA) += K * 0.5;   M(jA, iA) += K * 0.5;
                    M(iB, jB) += K * 0.5;   M(jB, iB) += K * 0.5;
                }

                // T13: coupling (l,m,A/B) ↔ (l+1, m+1, A/B)
                {
                    double K = -a * pf * inv_sqrt2
                              * std::sqrt((double)(l + m + 1) * (l + m + 2));
                    int jA = idx_map.at({l+1, m+1, 0});
                    int jB = idx_map.at({l+1, m+1, 1});
                    M(iA, jA) += K * 0.5;   M(jA, iA) += K * 0.5;
                    M(iB, jB) += K * 0.5;   M(jB, iB) += K * 0.5;
                }
            }

            // ── Beta XY: A_{l,m} ↔ B_{l, m±1} ─────────────────────────────

            // T21: K = -i · b_pf · √(½(l+m)(l-m+1))
            if (m > -l) {
                double mag = b_pf * std::sqrt(0.5 * (l + m) * (l - m + 1));
                cdouble K  = cdouble(0.0, -1.0) * mag;
                int jB = idx_map.at({l, m - 1, 1});
                M(iA, jB) += K * 0.5;
                M(jB, iA) += std::conj(K) * 0.5;
            }

            // T23: K = +i · b_pf · √(½(l-m)(l+m+1))
            if (m < l) {
                double mag = b_pf * std::sqrt(0.5 * (l - m) * (l + m + 1));
                cdouble K  = cdouble(0.0, +1.0) * mag;
                int jB = idx_map.at({l, m + 1, 1});
                M(iA, jB) += K * 0.5;
                M(jB, iA) += std::conj(K) * 0.5;
            }
        }
    }
    return M;
}

// ---- Result type ------------------------------------------------------------
struct Result {
    double               lambda_max;   // c† M c  under the chosen constraint
    double               norm_sq;      // ||c||²  (sanity check)
    double               power;        // dE/dt   (sanity check)
    std::vector<cdouble> c_opt;
    IndexMap             idx_map;
    int                  N;
};

// ============================================================================
// solve_max_xy_force
//
// Finds the unit-norm eigenvector/value first, then scales according to
// the chosen constraint so that the returned c_opt satisfies it exactly
// and lambda_max = c† M c  under that constraint.
//
// Constraint::Norm  => ||c||² = l_max  => scale by sqrt(l_max)
// Constraint::Power => dE/dt = P
//                   => ||c||² = 4π Ω² P / r²  => scale accordingly
// ============================================================================
Result solve_max_force(int l_max,
                       double r          = 1.0,
                       double Omega      = 1.0,
                       ConstraintType ct = ConstraintType::Norm,
                       double P_target   = 1.0,
                       bool verbose      = true) {
    int N = 0;
    IndexMap idx_map = build_index_map(l_max, N);

    MatrixXcd Mx = build_M_xy(l_max, r, Omega, idx_map, N);

    // Sanity-check Hermiticity
    double herm_err = (Mx - Mx.adjoint()).norm() / (Mx.norm() + 1e-300);
    if (herm_err > 1e-10)
        throw std::runtime_error("M_x is not Hermitian (rel err="
                                 + std::to_string(herm_err) + ")");

    // SelfAdjointEigenSolver returns eigenvalues in ascending order
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(Mx);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("Eigen solver failed");

    // Unit-norm optimal vector and its Rayleigh quotient
    double   lambda_unit = solver.eigenvalues()(N - 1);
    VectorXcd top        = solver.eigenvectors().col(N - 1);
    std::vector<cdouble> c_unit(top.data(), top.data() + N);

    // ---- Determine the norm prescribed by the chosen constraint ----
    //
    //  Norm  constraint:  ||c||² = l_max
    //  Power constraint:  dE/dt  = P_target
    //                     dE/dt  = (r²)/(4π Ω²) · ||c||²
    //                  => ||c||² = 4π Ω² P_target / r²
    double norm_sq_target = 0.0;
    if (ct == ConstraintType::Norm) {
        norm_sq_target = static_cast<double>(l_max);
    } else {   // Power
        norm_sq_target = 4.0 * PI * Omega * Omega * P_target / (r * r);
    }

    double scale = std::sqrt(norm_sq_target);   // ||c_unit||=1, so scale directly

    // ---- Build the scaled optimal vector ----
    std::vector<cdouble> c_opt(N);
    for (int i = 0; i < N; ++i) c_opt[i] = c_unit[i] * scale;

    // c† M c  = lambda_unit · ||c||²  (M is Hermitian, c_unit is its eigenvector)
    double lambda_max = lambda_unit * norm_sq_target;

    double actual_norm_sq = norm_sq_target;   // exact by construction
    double actual_power   = (r * r) / (4.0 * PI * Omega * Omega) * actual_norm_sq;

    // ---- Verbose output ----
    std::cout << "\n";
    std::cout << "Constraint     : "
              << (ct == ConstraintType::Norm ? "||c||² = l_max" : "dE/dt = P")
              << "\n";
    if (ct == ConstraintType::Power)
        std::cout << "Target dE/dt   : " << std::setprecision(10) << P_target << "\n";
    std::cout << "l_max          = " << l_max        << "\n"
              << "N              = " << N            << "\n"
              << "lambda_max     = " << std::setprecision(10) << lambda_max   << "\n"
              << "||c||²         = " << actual_norm_sq << "\n"
              << "dE/dt          = " << actual_power   << "\n\n";

    std::cout << std::setw(18) << "(l,m,type)"
              << std::setw(16) << "Re(c)"
              << std::setw(16) << "Im(c)"
              << std::setw(14) << "|c|" << "\n"
              << std::string(64, '-') << "\n";
    if (verbose) {
        for (auto& [key, idx] : idx_map) {
            cdouble val = c_opt[idx];
            if (std::abs(val) < 1e-8) continue;
            std::cout << "  (" << key.l << "," << std::showpos << key.m
                      << std::noshowpos << "," << (key.t == 0 ? 'A' : 'B') << ")"
                      << std::setw(16) << std::fixed << std::setprecision(6) << val.real()
                      << std::setw(16) << val.imag()
                      << std::setw(14) << std::abs(val) << "\n";
        }
    }

    return {lambda_max, actual_norm_sq, actual_power, c_opt, idx_map, N};
}

// ============================================================================
// save_coefs  –  writes CSV with header row indicating the constraint used
// ============================================================================
void save_coefs(const Result& res, int l_max, ConstraintType ct,
                double P_target, std::string folder) {
    mkdir(folder.c_str(), 0755);   // no-op if already exists

    int P = (int)P_target;

    std::string suffix = (ct == ConstraintType::Norm) ? "norm" : "power";
    std::string fname  = folder + "/maximized_coefs_xy_lMax_" + std::to_string(l_max)
                       + "_Power_" + std::to_string(P)
                       + "_" + suffix + ".csv";
    std::ofstream f(fname);
    if (!f) throw std::runtime_error("Cannot open " + fname);

    std::vector<double> A_re, A_im, B_re, B_im;
    std::vector<int>    Ls, Ms;

    for (auto& [key, idx] : res.idx_map) {
        if (key.t != 0) continue;
        cdouble A = res.c_opt[res.idx_map.at({key.l, key.m, 0})];
        cdouble B = res.c_opt[res.idx_map.at({key.l, key.m, 1})];
        if (std::abs(A) + std::abs(B) > 1e-8) {
            A_re.push_back(A.real()); A_im.push_back(A.imag());
            B_re.push_back(B.real()); B_im.push_back(B.imag());
            Ls.push_back(key.l);      Ms.push_back(key.m);
        }
    }

    f << std::fixed << std::setprecision(12);

    auto row = [&](const std::string& lbl, const auto& v) {
        f << lbl;
        for (auto x : v) f << "," << x;
        f << "\n";
    };
    row("A_re", A_re); row("A_im", A_im);
    row("B_re", B_re); row("B_im", B_im);
    row("M",    Ms);   row("L",    Ls);

    // Write metadata as commented header rows
    f << "# constraint," << (ct == ConstraintType::Norm ? "norm" : "power") << "\n";
    f << "# norm_sq,"    << res.norm_sq   << "\n";
    f << "# dEdt,"       << res.power     << "\n";
    f << "# lambda_max," << res.lambda_max << "\n";

    std::cout << "Saved to " << fname << "\n";
}

// ---- main -------------------------------------------------------------------
int main(int argc, char* argv[]) {
    int    l_max  = (argc > 1) ? std::stoi(argv[1]) : 10;
    int    c      = (argc > 2) ? std::stoi(argv[2]) : 1;   // 0=power, 1=norm
    double P_target = (argc > 3) ? std::stod(argv[3]) : 1.0;
    double r      = (argc > 4) ? std::stod(argv[4]) : 1.0;
    double Omega  = (argc > 5) ? std::stod(argv[5]) : 1.0;
    std::string folder = (argc > 6) ? argv[6] : "csvs";

    ConstraintType ct = (c == 0) ? ConstraintType::Power : ConstraintType::Norm;

    std::cout << "=== Multipole XY-Force Maximization ===\n"
              << "l_max=" << l_max << "  r=" << r << "  Omega=" << Omega << "\n";

    Result res = solve_max_force(l_max, r, Omega, ct, P_target, /*verbose=*/false);
    save_coefs(res, l_max, ct, P_target, folder);
    return 0;
}