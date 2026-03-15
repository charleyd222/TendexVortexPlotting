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
 * Compile (macOS/Linux):
 *   clang++ -std=c++17 -O3 -march=native multipole_opt.cpp -o multipole_opt \
 *       -I /usr/local/include/eigen3
 *
 * Usage:
 *   ./multipole_opt <l_max> [r] [Omega] [--constraint norm|power] [--power P]
 *
 * Examples:
 *   ./multipole_opt 10                               # norm constraint, l_max=10
 *   ./multipole_opt 10 1.0 1.0 --constraint power    # power=1.0 (default)
 *   ./multipole_opt 10 1.0 1.0 --constraint power --power 2.5
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
#include <sys/stat.h>   // add at the top with other includes


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
    double               lambda_max;   // largest unit-norm eigenvalue
    std::vector<cdouble> top_eigvec;   // corresponding eigenvector (block-local)
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

        // alpha coupling: (l,A) <-> (l+1,A)  and  (l,B) <-> (l+1,B)
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

        // beta coupling: (l,A) <-> (l,B)  (purely imaginary, anti-Hermitian off-diag)
        cdouble beta = cdouble(0.0, -1.0) * (double)m
                     / (8.0 * PI * l * (l + 1)) * prefactor;
        A(iA, iB) += beta            * 0.5;
        A(iB, iA) += std::conj(beta) * 0.5;
    }

    // SelfAdjointEigenSolver returns eigenvalues in ascending order
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(A);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("Eigen solver failed for m=" + std::to_string(m));

    VectorXcd top = solver.eigenvectors().col(n - 1);
    std::vector<cdouble> top_vec(top.data(), top.data() + n);
    return {solver.eigenvalues()(n - 1), top_vec};
}

// ============================================================================
// solve_max_force
//
// Finds the unit-norm eigenvector/value first, then scales according to
// the chosen constraint so that the returned c_opt satisfies it exactly
// and lambda_max = c† M c  under that constraint.
//
// Constraint::Norm  => ||c||² = l_max  => scale by sqrt(l_max)
// Constraint::Power => dE/dt = P
//                   => ||c||² = 4π Ω² P / r²  => scale accordingly
// ============================================================================
struct Result {
    double               lambda_max;   // c† M c  under the chosen constraint
    double               norm_sq;      // ||c||²  (sanity check)
    double               power;        // dE/dt   (sanity check)
    std::vector<cdouble> c_opt;
    IndexMap             idx_map;
    int                  N;
};

Result solve_max_force(int l_max,
                       double r          = 1.0,
                       double Omega      = 1.0,
                       ConstraintType ct = ConstraintType::Norm,
                       double P_target   = 1.0,   // only used for Power constraint
                       bool verbose      = true)
{
    int N = 0;
    IndexMap idx_map = build_index_map(l_max, N);
    std::vector<cdouble> c_unit(N, 0.0);   // will hold the unit-norm optimal c
    double lambda_unit = -1e300;            // eigenvalue for ||c||=1

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

    // Compute actual dE/dt for the output record
    double actual_norm_sq = norm_sq_target;   // exact by construction
    double actual_power   = (r * r) / (4.0 * PI * Omega * Omega) * actual_norm_sq;

    // ---- Verbose output ----
    //if (verbose) {
        std::cout << "\n";
        std::cout << "Constraint     : "
                  << (ct == ConstraintType::Norm ? "||c||² = l_max" : "dE/dt = P")
                  << "\n";
        if (ct == ConstraintType::Power)
            std::cout << "Target dE/dt   : " << std::setprecision(10) << P_target << "\n";
        std::cout << "l_max          = " << l_max     << "\n"
                  << "N              = " << N         << "\n"
                  << "lambda_max     = " << std::setprecision(10) << lambda_max  << "\n"
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
void save_coefs(const Result& res, int l_max, ConstraintType ct, double P_target, std::string folder) {
    //const std::string folder = "csvs";
    mkdir(folder.c_str(), 0755);   // no-op if already exists

    int P = (int)P_target;

    std::string suffix = (ct == ConstraintType::Norm)
                       ? "norm"
                       : "power";
    std::string fname = folder + "/maximized_coefs_lMax_" + std::to_string(l_max) + "_Power_" + std::to_string(P)
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
    f << "# norm_sq,"    << res.norm_sq  << "\n";
    f << "# dEdt,"       << res.power    << "\n";
    f << "# lambda_max," << res.lambda_max << "\n";

    std::cout << "Saved to " << fname << "\n";
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char* argv[]) {
    // --- Parse positional args ---
    int l_max = (argc > 1) ? std::stoi(argv[1]) : 10;
    int c = (argc > 2) ? std::stod(argv[2]) : 1.0; // 0 for P 1 for lmax
    double P_target = (argc > 3) ? std::stod(argv[3]) : 1.0;
    double r     = (argc > 4) ? std::stod(argv[4]) : 1.0;
    double Omega = (argc > 5) ? std::stod(argv[5]) : 1.0;
    std::string folder = (argc > 6) ? argv[6] : "csvs";

    // --- Parse optional flags ---
    ConstraintType ct = ConstraintType::Norm;

    if (c == 1) ct = ConstraintType::Norm;
    else if (c == 0) ct = ConstraintType::Power;

    std::cout << "=== Multipole Force Maximization ===\n"
              << "l_max=" << l_max << "  r=" << r << "  Omega=" << Omega << "\n";

    Result res = solve_max_force(l_max, r, Omega, ct, P_target, /*verbose=*/false);
    save_coefs(res, l_max, ct, P_target, folder);
    return 0;
}