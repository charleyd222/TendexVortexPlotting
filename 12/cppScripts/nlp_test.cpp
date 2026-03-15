/*
 * multipole_opt.cpp
 *
 * Maximises  dP_z/dt = c† M c  subject to one of two constraints:
 *
 *   --constraint norm   ||c||² = l_max                          (default)
 *
 *   --constraint power  dE/dt = P  (uniform-power constraint)
 *       where  dE/dt = (1/32π) Σ_{l,m} (8r²/Ω²)(|A^{lm}|² + |B^{lm}|²)
 *                     = (r²)/(4π Ω²) · ||c||²
 *       so fixing dE/dt = P  ⟺  ||c||² = 4π Ω² P / r²
 *       The optional --power <P> flag sets the target power (default P=1).
 *
 * KEY INSIGHT: M is block-diagonal in m.  Each m-sector couples only
 * l-values with the same m, so the full NxN problem decomposes into
 * (2*l_max+1) independent blocks, each of size ≤ 2*(l_max-|m|+1).
 * We diagonalise each block exactly with Eigen's SelfAdjointEigenSolver,
 * then take the global maximum eigenvalue across all m-sectors.
 * This exactly reproduces scipy.linalg.eigh on the full matrix.
 *
 * PER-ELL NORMALISATION:
 *   After the global optimisation, each ℓ-block is independently rescaled
 *   so that  Σ_{m,t} |c_{l,m,t}|² = 1  for every l ∈ [2, l_max].
 *   Blocks whose raw norm is zero (no active coefficients at that ℓ) are
 *   left as zero rather than divided by zero.
 *
 * Compile (macOS/Linux):
 *   clang++ -std=c++17 -O3 -march=native multipole_opt.cpp -o multipole_opt \
 *       -I /usr/local/include/eigen3
 *
 * Usage:
 *   ./multipole_opt <l_max> [c] [P_target] [r] [Omega]
 *       c = 1  →  norm  constraint (default)
 *       c = 0  →  power constraint
 *
 * Examples:
 *   ./multipole_opt 10                               # norm constraint, l_max=10
 *   ./multipole_opt 10 0 1.0                         # power=1.0
 *   ./multipole_opt 10 0 2.5 1.0 1.0                 # power=2.5, r=1, Ω=1
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

// ============================================================================
// Index map  (l,m,t) -> flat index
// ============================================================================
struct Key {
    int l, m, t;
    bool operator<(const Key& o) const {
        if (l != o.l) return l < o.l;
        if (m != o.m) return m < o.m;
        return t < o.t;
    }
};
using IndexMap = std::map<Key, int>;

IndexMap build_index_map(int l_max, int& N) {
    IndexMap idx;
    int i = 0;
    for (int l = 2; l <= l_max; ++l)
        for (int m = -l; m <= l; ++m) {
            idx[{l, m, 0}] = i++;   // A^{lm}
            idx[{l, m, 1}] = i++;   // B^{lm}
        }
    N = i;
    return idx;
}

// ============================================================================
// Build and diagonalise one m-block
// ============================================================================
struct BlockResult {
    double               lambda_max;
    std::vector<cdouble> top_eigvec;
};

BlockResult solve_block(int m, int l_max, double r, double Omega) {
    int l_min = std::max(2, std::abs(m));
    if (l_min > l_max) return {-1e300, {}};

    int n = 2 * (l_max - l_min + 1);
    if (n == 0) return {-1e300, {}};

    MatrixXcd A = MatrixXcd::Zero(n, n);

    double prefactor = 8.0 * r * r / (Omega * Omega);

    for (int l = l_min; l <= l_max; ++l) {
        int iA = 2 * (l - l_min);
        int iB = 2 * (l - l_min) + 1;

        if (l + 1 <= l_max) {
            int iA1 = 2 * (l + 1 - l_min);
            int iB1 = 2 * (l + 1 - l_min) + 1;

            double a_coef = 1.0 / (32.0 * PI * (l + 1))
                          * std::sqrt((2.0 * (l - 1) * (l + 3))
                                      / ((2.0 * l + 1) * (2.0 * l + 3)));
            double alpha  = a_coef
                          * std::sqrt(2.0 * (l - m + 1) * (l + m + 1))
                          * prefactor;

            A(iA,  iA1) += alpha * 0.5;
            A(iA1, iA ) += alpha * 0.5;
            A(iB,  iB1) += alpha * 0.5;
            A(iB1, iB ) += alpha * 0.5;
        }

        cdouble beta = cdouble(0.0, -1.0) * (double)m
                     / (8.0 * PI * l * (l + 1)) * prefactor;
        A(iA, iB) += beta            * 0.5;
        A(iB, iA) += std::conj(beta) * 0.5;
    }

    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(A);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("Eigen solver failed for m=" + std::to_string(m));

    VectorXcd top = solver.eigenvectors().col(n - 1);
    std::vector<cdouble> top_vec(top.data(), top.data() + n);
    return {solver.eigenvalues()(n - 1), top_vec};
}

// ============================================================================
// normalize_per_ell
//
// For each l in [2, l_max], rescales c so that
//   Σ_{m=-l}^{l}  ( |A^{lm}|² + |B^{lm}|² )  =  1
//
// Returns the per-ell norms *before* normalisation (useful for diagnostics).
// Blocks with zero norm are left untouched (all-zero stays all-zero).
// ============================================================================
std::vector<double> normalize_per_ell(std::vector<cdouble>& c,
                                      const IndexMap& idx_map,
                                      int l_max)
{
    std::vector<double> norms(l_max + 1, 0.0);

    // --- Pass 1: compute per-ell norm² ---
    for (auto& [key, idx] : idx_map)
        norms[key.l] += std::norm(c[idx]);   // std::norm = |z|²

    // --- Pass 2: rescale ---
    for (auto& [key, idx] : idx_map) {
        double n = std::sqrt(norms[key.l]);
        if (n > 1e-15) c[idx] /= n;
    }

    // Return the pre-normalisation norms (not norm²) for reporting
    for (int l = 2; l <= l_max; ++l)
        norms[l] = std::sqrt(norms[l]);

    return norms;   // norms[l] = ||c_l||  before normalisation
}

// ============================================================================
// Result struct
// ============================================================================
struct Result {
    double               lambda_max;   // c† M c  under the chosen constraint
    double               norm_sq;      // ||c||²  (full vector, post-normalisation)
    double               power;        // dE/dt   (post-normalisation)
    std::vector<cdouble> c_opt;        // per-ell-normalised optimal coefficients
    std::vector<double>  ell_norms;    // pre-normalisation per-ell norms
    IndexMap             idx_map;
    int                  N;
};

// ============================================================================
// solve_max_force
// ============================================================================
Result solve_max_force(int l_max,
                       double r          = 1.0,
                       double Omega      = 1.0,
                       ConstraintType ct = ConstraintType::Norm,
                       double P_target   = 1.0,
                       bool verbose      = true)
{
    int N = 0;
    IndexMap idx_map = build_index_map(l_max, N);
    std::vector<cdouble> c_unit(N, 0.0);
    double lambda_unit = -1e300;

    // ---- Find the best m-block ----
    for (int m = -l_max; m <= l_max; ++m) {
        BlockResult br = solve_block(m, l_max, r, Omega);
        if (br.lambda_max <= lambda_unit) continue;

        lambda_unit = br.lambda_max;
        std::fill(c_unit.begin(), c_unit.end(), 0.0);

        int l_min = std::max(2, std::abs(m));
        for (int l = l_min; l <= l_max; ++l) {
            c_unit[idx_map.at({l, m, 0})] = br.top_eigvec[2 * (l - l_min)];
            c_unit[idx_map.at({l, m, 1})] = br.top_eigvec[2 * (l - l_min) + 1];
        }
    }

    // ---- Scale to satisfy the chosen constraint ----
    double norm_sq_target = 0.0;
    if (ct == ConstraintType::Norm) {
        norm_sq_target = static_cast<double>(l_max);
    } else {
        norm_sq_target = 4.0 * PI * Omega * Omega * P_target / (r * r);
    }

    double scale = std::sqrt(norm_sq_target);
    std::vector<cdouble> c_opt(N);
    for (int i = 0; i < N; ++i) c_opt[i] = c_unit[i] * scale;

    double lambda_max   = lambda_unit * norm_sq_target;
    double actual_power = (r * r) / (4.0 * PI * Omega * Omega) * norm_sq_target;

    // ---- Per-ell normalisation ----
    // Record raw per-ell norms, then rescale each ℓ-block to unit norm.
    std::vector<double> ell_norms = normalize_per_ell(c_opt, idx_map, l_max);

    // Recompute ||c||² and dE/dt after per-ell normalisation
    double actual_norm_sq = 0.0;
    for (auto& z : c_opt) actual_norm_sq += std::norm(z);
    double actual_power_norm = (r * r) / (4.0 * PI * Omega * Omega) * actual_norm_sq;

    // ---- Verbose output ----
    if (verbose) {
        std::cout << "\n";
        std::cout << "Constraint     : "
                  << (ct == ConstraintType::Norm ? "||c||² = l_max" : "dE/dt = P")
                  << "\n";
        if (ct == ConstraintType::Power)
            std::cout << "Target dE/dt   : " << std::setprecision(10) << P_target << "\n";
        std::cout << "l_max          = " << l_max      << "\n"
                  << "N              = " << N          << "\n"
                  << "lambda_max     = " << std::setprecision(10) << lambda_max  << "\n"
                  << "(before per-ell norm)\n"
                  << "  ||c||²       = " << norm_sq_target  << "\n"
                  << "  dE/dt        = " << actual_power    << "\n"
                  << "(after  per-ell norm)\n"
                  << "  ||c||²       = " << actual_norm_sq      << "\n"
                  << "  dE/dt        = " << actual_power_norm   << "\n\n";

        // Per-ell norm summary
        std::cout << std::setw(8)  << "ell"
                  << std::setw(20) << "pre-norm ||c_l||"
                  << std::setw(20) << "post-norm ||c_l||" << "\n"
                  << std::string(50, '-') << "\n";
        for (int l = 2; l <= l_max; ++l) {
            // post-norm should be 1 if the block was non-zero, else 0
            double post = (ell_norms[l] > 1e-15) ? 1.0 : 0.0;
            std::cout << std::setw(8) << l
                      << std::setw(20) << std::fixed << std::setprecision(6) << ell_norms[l]
                      << std::setw(20) << post << "\n";
        }
        std::cout << "\n";

        // Full coefficient table
        std::cout << std::setw(18) << "(l,m,type)"
                  << std::setw(16) << "Re(c)"
                  << std::setw(16) << "Im(c)"
                  << std::setw(14) << "|c|" << "\n"
                  << std::string(64, '-') << "\n";

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

    return {lambda_max, actual_norm_sq, actual_power_norm, c_opt,
            ell_norms, idx_map, N};
}

// ============================================================================
// save_coefs
// ============================================================================
void save_coefs(const Result& res, int l_max, ConstraintType ct, double P_target) {
    const std::string folder = "csvs";
    mkdir(folder.c_str(), 0755);

    int P = (int)P_target;

    std::string suffix = (ct == ConstraintType::Norm) ? "norm" : "power";
    std::string fname  = folder + "/maximized_coefs_lMax_" + std::to_string(l_max)
                       + "_Power_" + std::to_string(P)
                       + "_" + suffix + "_per_ell_norm.csv";
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

    // Per-ell norms (before normalisation) as metadata
    f << "# per_ell_pre_norm";
    for (int l = 2; l <= l_max; ++l) f << "," << res.ell_norms[l];
    f << "\n";

    f << "# constraint," << (ct == ConstraintType::Norm ? "norm" : "power") << "\n";
    f << "# norm_sq,"    << res.norm_sq   << "\n";
    f << "# dEdt,"       << res.power     << "\n";
    f << "# lambda_max," << res.lambda_max << "\n";
    f << "# note,coefficients_normalised_per_ell_(sum_|c_lmt|^2_over_m_t_equals_1)\n";

    std::cout << "Saved to " << fname << "\n";
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char* argv[]) {
    int    l_max    = (argc > 1) ? std::stoi(argv[1]) : 10;
    int    c        = (argc > 2) ? std::stoi(argv[2]) : 1;   // 1=norm, 0=power
    double P_target = (argc > 3) ? std::stod(argv[3]) : 1.0;
    double r        = (argc > 4) ? std::stod(argv[4]) : 1.0;
    double Omega    = (argc > 5) ? std::stod(argv[5]) : 1.0;

    ConstraintType ct = (c == 0) ? ConstraintType::Power : ConstraintType::Norm;

    std::cout << "=== Multipole Force Maximization (per-ell normalised) ===\n"
              << "l_max=" << l_max << "  r=" << r << "  Omega=" << Omega << "\n";

    Result res = solve_max_force(l_max, r, Omega, ct, P_target, /*verbose=*/true);
    save_coefs(res, l_max, ct, P_target);
    return 0;
}