/*
 * multipole_opt_xyz.cpp
 *
 * Maximises the magnitude of the total force:
 *   |F| = sqrt(F_x² + F_y² + F_z²),   with F_i = c† M_i c
 *
 * Using the power constraint from nlp.cpp:
 *   dE/dt = P  (uniform-power constraint)
 *   where  dE/dt = (r²)/(4π Ω²) · ||c||²
 *   so fixing dE/dt = P  ⟺  ||c||² = 4π Ω² P / r²
 *
 * The force components F_x, F_y, F_z couple across different (l,m) modes,
 * so each M_i is a general (non-block-diagonal) N×N Hermitian matrix.
 * Because |F| is nonlinear in c, we solve via fixed-point directional
 * optimization over u·F(c), where u is a unit vector in R^3.
 *
 * Compile (macOS/Linux):
 *   clang++ -std=c++17 -O3 -march=native nlp_xyz.cpp -o nlp_xyz \
 *       -I /usr/local/include/eigen3
 *
 * Usage:
 *   ./nlp_xyz <l_max> [c] [P_target] [r] [Omega] [folder]
 *       c=0 → power constraint,  c=1 → norm constraint (default)
 *
 * Examples:
 *   ./nlp_xyz 10                                # norm constraint, l_max=10
 *   ./nlp_xyz 10 0 1.0                         # power constraint, P=1
 *   ./nlp_xyz 10 0 2.5 1.0 1.0 csvs           # power=2.5, r=1, Ω=1
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
#include <array>
#include <random>
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

// ============================================================================
// Build M_z: Z-component force matrix (from nlp.cpp formulation)
// ============================================================================
// F_z couples (l,m) ↔ (l+1,m) via alpha (diagonal in m)
// and couples (l,m,A) ↔ (l,m,B) via beta
//
MatrixXcd build_M_z(int l_max, double r, double Omega, 
                    const IndexMap& idx_map, int N) {
    MatrixXcd M = MatrixXcd::Zero(N, N);
    double pf = 8.0 * r * r / (Omega * Omega);

    for (int l = 2; l <= l_max; ++l) {
        for (int m = -l; m <= l; ++m) {
            int iA = idx_map.at({l, m, 0});
            int iB = idx_map.at({l, m, 1});

            // ── Alpha_z: l ↔ l+1, m stays the same ───────────────────────
            if (l + 1 <= l_max) {
                double a = 1.0 / (32.0 * PI * (l + 1))
                         * std::sqrt((2.0 * (l - 1) * (l + 3))
                                     / ((2.0 * l + 1) * (2.0 * l + 3)));
                double alpha = a * std::sqrt(2.0 * (l - m + 1) * (l + m + 1)) * pf;

                int jA = idx_map.at({l + 1, m, 0});
                int jB = idx_map.at({l + 1, m, 1});

                M(iA, jA) += alpha * 0.5;  M(jA, iA) += alpha * 0.5;
                M(iB, jB) += alpha * 0.5;  M(jB, iB) += alpha * 0.5;
            }

            // ── Beta_z: A_{l,m} ↔ B_{l,m} at same (l,m) ──────────────────
            // beta = (-i*m / (8π*l*(l+1))) * (8r²/Ω²)
            cdouble beta = cdouble(0.0, -1.0) * (double)m
                         / (8.0 * PI * l * (l + 1)) * pf;
            M(iA, iB) += beta * 0.5;
            M(iB, iA) += std::conj(beta) * 0.5;
        }
    }
    return M;
}

// ============================================================================
// Build M_x and M_y: XY-component force matrices (from nlp_xy.cpp formulation)
// ============================================================================
// Both F_x and F_y couple (l,m) ↔ (l+1,m±1) via alpha (different in m)
// and (l,m,A) ↔ (l,m±1,B) via beta terms.
// They are related by a 90° rotation, so we build M_x and handle M_y similarly.
//
MatrixXcd build_M_x(int l_max, double r, double Omega,
                    const IndexMap& idx_map, int N) {
    MatrixXcd M = MatrixXcd::Zero(N, N);
    double pf = 8.0 * r * r / (Omega * Omega);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    for (int l = 2; l <= l_max; ++l) {
        double b_pf = pf / (4.0 * PI * l * (l + 1));

        for (int m = -l; m <= l; ++m) {
            int iA = idx_map.at({l, m, 0});
            int iB = idx_map.at({l, m, 1});

            // ── Alpha_x: l ↔ l+1, m shifts by ±1 ──────────────────────────
            if (l + 1 <= l_max) {
                double a = 1.0 / (32.0 * PI * (l + 1))
                         * std::sqrt((2.0 * (l - 1) * (l + 3))
                                     / ((2.0 * l + 1) * (2.0 * l + 3)));

                // T11: coupling (l,m,A/B) ↔ (l+1, m-1, A/B)
                {
                    double K = a * pf * inv_sqrt2
                             * std::sqrt((double)(l - m + 1) * (l - m + 2));
                    int jA = idx_map.at({l + 1, m - 1, 0});
                    int jB = idx_map.at({l + 1, m - 1, 1});
                    M(iA, jA) += K * 0.5;   M(jA, iA) += K * 0.5;
                    M(iB, jB) += K * 0.5;   M(jB, iB) += K * 0.5;
                }

                // T13: coupling (l,m,A/B) ↔ (l+1, m+1, A/B)
                {
                    double K = -a * pf * inv_sqrt2
                              * std::sqrt((double)(l + m + 1) * (l + m + 2));
                    int jA = idx_map.at({l + 1, m + 1, 0});
                    int jB = idx_map.at({l + 1, m + 1, 1});
                    M(iA, jA) += K * 0.5;   M(jA, iA) += K * 0.5;
                    M(iB, jB) += K * 0.5;   M(jB, iB) += K * 0.5;
                }
            }

            // ── Beta_x: A_{l,m} ↔ B_{l, m±1} ─────────────────────────────
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

// Build M_y similarly (with sign changes and factors of i for y-component)
MatrixXcd build_M_y(int l_max, double r, double Omega,
                    const IndexMap& idx_map, int N) {
    MatrixXcd M = MatrixXcd::Zero(N, N);
    double pf = 8.0 * r * r / (Omega * Omega);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    for (int l = 2; l <= l_max; ++l) {
        double b_pf = pf / (4.0 * PI * l * (l + 1));

        for (int m = -l; m <= l; ++m) {
            int iA = idx_map.at({l, m, 0});
            int iB = idx_map.at({l, m, 1});

            // ── Alpha_y: l ↔ l+1, m shifts by ±1 (with -i factor) ────────
            if (l + 1 <= l_max) {
                double a = 1.0 / (32.0 * PI * (l + 1))
                         * std::sqrt((2.0 * (l - 1) * (l + 3))
                                     / ((2.0 * l + 1) * (2.0 * l + 3)));

                // T11y: coupling with -i factor for y
                {
                    double K_mag = a * pf * inv_sqrt2
                                * std::sqrt((double)(l - m + 1) * (l - m + 2));
                    cdouble K = cdouble(0.0, -1.0) * K_mag;
                    int jA = idx_map.at({l + 1, m - 1, 0});
                    int jB = idx_map.at({l + 1, m - 1, 1});
                    M(iA, jA) += K * 0.5;   M(jA, iA) += std::conj(K) * 0.5;
                    M(iB, jB) += K * 0.5;   M(jB, iB) += std::conj(K) * 0.5;
                }

                // T13y: coupling with -i factor for y
                {
                    double K_mag = a * pf * inv_sqrt2
                                * std::sqrt((double)(l + m + 1) * (l + m + 2));
                    cdouble K = cdouble(0.0, +1.0) * K_mag;
                    int jA = idx_map.at({l + 1, m + 1, 0});
                    int jB = idx_map.at({l + 1, m + 1, 1});
                    M(iA, jA) += K * 0.5;   M(jA, iA) += std::conj(K) * 0.5;
                    M(iB, jB) += K * 0.5;   M(jB, iB) += std::conj(K) * 0.5;
                }
            }

            // ── Beta_y: A_{l,m} ↔ B_{l, m±1} (with real coefficients for y)
            // T21y: K = b_pf · √(½(l+m)(l-m+1))
            if (m > -l) {
                double K = b_pf * std::sqrt(0.5 * (l + m) * (l - m + 1));
                int jB = idx_map.at({l, m - 1, 1});
                M(iA, jB) += -K * 0.5;
                M(jB, iA) += -K * 0.5;
            }

            // T23y: K = b_pf · √(½(l-m)(l+m+1))
            if (m < l) {
                double K = b_pf * std::sqrt(0.5 * (l - m) * (l + m + 1));
                int jB = idx_map.at({l, m + 1, 1});
                M(iA, jB) += K * 0.5;
                M(jB, iA) += K * 0.5;
            }
        }
    }
    return M;
}

// ---- Result type ------------------------------------------------------------
struct Result {
    double               lambda_max;   // |F| under the chosen constraint
    double               force_x;
    double               force_y;
    double               force_z;
    double               norm_sq;      // ||c||²  (sanity check)
    double               power;        // dE/dt   (sanity check)
    int                  seed_count;
    int                  best_iterations;
    double               best_du;
    double               spectral_bound_easy;    // Easy bound: s·√(||M_x||_2² + ||M_y||_2² + ||M_z||_2²)
    double               spectral_bound_tight;   // Tight bound: s·√(λ_max(M_x² + M_y² + M_z²))
    std::vector<cdouble> c_opt;
    IndexMap             idx_map;
    int                  N;
};

double quadratic_real(const std::vector<cdouble>& c, const MatrixXcd& M) {
    Eigen::Map<const VectorXcd> v(c.data(), (Eigen::Index)c.size());
    cdouble q = v.dot(M * v);
    return q.real();
}

void largest_eigenpair(const MatrixXcd& M,
                       Eigen::SelfAdjointEigenSolver<MatrixXcd>& solver,
                       VectorXcd& top_vec,
                       double& top_eval)
{
    solver.compute(M, Eigen::ComputeEigenvectors);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("Eigen solver failed in directional step");

    Eigen::Index n = M.rows();
    top_eval = solver.eigenvalues()(n - 1);
    top_vec = solver.eigenvectors().col(n - 1);
}

// ============================================================================
// solve_max_xyz_force
//
// Finds a unit-norm maximizer of |F| = sqrt(Fx^2 + Fy^2 + Fz^2)
// by fixed-point directional optimization, then scales according
// to the chosen constraint.
//
// Constraint::Norm  => ||c||² = l_max
// Constraint::Power => dE/dt = P  => ||c||² = 4π Ω² P / r²
// ============================================================================
Result solve_max_force(int l_max,
                       double r          = 1.0,
                       double Omega      = 1.0,
                       ConstraintType ct = ConstraintType::Norm,
                       double P_target   = 1.0,
                       bool verbose      = true) {
    int N = 0;
    IndexMap idx_map = build_index_map(l_max, N);

    // Build all three component matrices
    MatrixXcd Mz = build_M_z(l_max, r, Omega, idx_map, N);
    MatrixXcd Mx = build_M_x(l_max, r, Omega, idx_map, N);
    MatrixXcd My = build_M_y(l_max, r, Omega, idx_map, N);

    auto check_hermitian = [](const MatrixXcd& M, const char* name) {
        double herm_err = (M - M.adjoint()).norm() / (M.norm() + 1e-300);
        if (herm_err > 1e-10)
            std::cerr << "Warning: " << name << " is not perfectly Hermitian (rel err="
                      << herm_err << ")\n";
    };
    check_hermitian(Mx, "M_x");
    check_hermitian(My, "M_y");
    check_hermitian(Mz, "M_z");

    // ---- Compute easy spectral bound (Horn & Johnson, Matrix Analysis) ----
    // Upper bound: max |F(c)| ≤ ||c||² · √(||M_x||_2² + ||M_y||_2² + ||M_z||_2²)
    // where ||M||_2 is the spectral norm (largest singular value = largest eigenvalue for Hermitian).
    Eigen::SelfAdjointEigenSolver<MatrixXcd> eig_Mx(Mx), eig_My(My), eig_Mz(Mz);
    double spectral_Mx = std::abs(eig_Mx.eigenvalues().maxCoeff());
    double spectral_My = std::abs(eig_My.eigenvalues().maxCoeff());
    double spectral_Mz = std::abs(eig_Mz.eigenvalues().maxCoeff());
    double spectral_bound_factor = std::sqrt(spectral_Mx*spectral_Mx 
                                            + spectral_My*spectral_My 
                                            + spectral_Mz*spectral_Mz);

    // ---- Compute tight spectral bound (Cauchy-Schwarz on eigenvectors) ----
    // Tighter bound using max eigenvalue of combined matrix M_x² + M_y² + M_z²:
    // |F(c)|² = (c† M_x c)² + (c† M_y c)² + (c† M_z c)²
    //         ≤ c† (M_x² + M_y² + M_z²) c  (Cauchy-Schwarz)
    // So max |F(c)| ≤ ||c||² · √(λ_max(M_x² + M_y² + M_z²))
    MatrixXcd M_combined = Mx*Mx + My*My + Mz*Mz;
    Eigen::SelfAdjointEigenSolver<MatrixXcd> eig_combined(M_combined);
    if (eig_combined.info() != Eigen::Success)
        std::cerr << "Warning: eigensolve for combined matrix failed\n";
    double tight_bound_factor = std::sqrt(std::abs(eig_combined.eigenvalues().maxCoeff()));

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

    // Compute the two spectral bounds
    double spectral_bound_easy = norm_sq_target * spectral_bound_factor;
    double spectral_bound_tight = norm_sq_target * tight_bound_factor;

    // Fixed-point directional optimization for unit-norm coefficients:
    // 1) for unit u, maximize u·F(c) = c†(u_x Mx + u_y My + u_z Mz)c
    // 2) update u <- F/|F|
    // 3) repeat from several initial directions; keep best |F|
    std::vector<std::array<double, 3>> seeds = {
        {{1.0, 0.0, 0.0}}, {{-1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}}, {{0.0, -1.0, 0.0}},
        {{0.0, 0.0, 1.0}}, {{0.0, 0.0, -1.0}},
        {{1.0, 1.0, 1.0}}, {{1.0, -1.0, 1.0}},
        {{-1.0, 1.0, 1.0}}, {{1.0, 1.0, -1.0}},
        {{-1.0, -1.0, 1.0}}, {{-1.0, 1.0, -1.0}},
        {{1.0, -1.0, -1.0}}
    };

    // Add random directions to improve robustness against local fixed points.
    constexpr int random_seed_count = 24;
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int i = 0; i < random_seed_count; ++i) {
        std::array<double, 3> u = {{nd(rng), nd(rng), nd(rng)}};
        double n = std::sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
        if (n < 1e-14) {
            u = {{1.0, 0.0, 0.0}};
            n = 1.0;
        }
        u[0] /= n;
        u[1] /= n;
        u[2] /= n;
        seeds.push_back(u);
    }

    VectorXcd best_c_unit_vec = VectorXcd::Zero(N);
    double best_mag_unit = -1.0;
    double best_fx_unit = 0.0, best_fy_unit = 0.0, best_fz_unit = 0.0;
    constexpr int max_iter = 200;
    constexpr double tol = 1e-12;
    int best_iterations = max_iter;
    double best_du = 1e300;

    MatrixXcd Mdir(N, N);
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(N);
    VectorXcd c_unit = VectorXcd::Zero(N);
    VectorXcd tmp(N);

    for (const auto& seed : seeds) {
        double u_norm = std::sqrt(seed[0]*seed[0] + seed[1]*seed[1] + seed[2]*seed[2]);
        std::array<double, 3> u = {{seed[0]/u_norm, seed[1]/u_norm, seed[2]/u_norm}};

        c_unit.setZero();
        double fx = 0.0, fy = 0.0, fz = 0.0;
        int it_used = 0;
        double du_last = 1e300;

        for (int it = 0; it < max_iter; ++it) {
            Mdir.noalias() = u[0] * Mx;
            Mdir += u[1] * My;
            Mdir += u[2] * Mz;
            double eig = 0.0;
            largest_eigenpair(Mdir, solver, c_unit, eig);

            tmp.noalias() = Mx * c_unit;
            fx = c_unit.dot(tmp).real();
            tmp.noalias() = My * c_unit;
            fy = c_unit.dot(tmp).real();
            tmp.noalias() = Mz * c_unit;
            fz = c_unit.dot(tmp).real();

            double mag = std::sqrt(fx*fx + fy*fy + fz*fz);
            if (mag < 1e-16) break;

            std::array<double, 3> u_new = {{fx / mag, fy / mag, fz / mag}};
            double du = std::sqrt((u_new[0]-u[0])*(u_new[0]-u[0])
                                + (u_new[1]-u[1])*(u_new[1]-u[1])
                                + (u_new[2]-u[2])*(u_new[2]-u[2]));
            du_last = du;
            it_used = it + 1;
            u = u_new;
            if (du < tol) break;
        }

        double mag_unit = std::sqrt(fx*fx + fy*fy + fz*fz);
        if (mag_unit > best_mag_unit) {
            best_mag_unit = mag_unit;
            best_fx_unit = fx;
            best_fy_unit = fy;
            best_fz_unit = fz;
            best_c_unit_vec = c_unit;
            best_iterations = it_used;
            best_du = du_last;
        }
    }

    double scale = std::sqrt(norm_sq_target);   // ||c_unit||=1, so scale directly

    // ---- Build the scaled optimal vector ----
    std::vector<cdouble> c_opt(N);
    for (int i = 0; i < N; ++i) c_opt[i] = best_c_unit_vec(i) * scale;

    // Each force component scales as ||c||², so |F| does too.
    double force_x = best_fx_unit * norm_sq_target;
    double force_y = best_fy_unit * norm_sq_target;
    double force_z = best_fz_unit * norm_sq_target;
    double lambda_max = best_mag_unit * norm_sq_target;

    double actual_norm_sq = norm_sq_target;   // exact by construction
    double actual_power   = (r * r) / (4.0 * PI * Omega * Omega) * actual_norm_sq;

    // ---- Verbose output ----
    std::cout << "\n=== XYZ Force Maximization ===\n";
    std::cout << "Constraint     : "
              << (ct == ConstraintType::Norm ? "||c||² = l_max" : "dE/dt = P")
              << "\n";
    if (ct == ConstraintType::Power)
        std::cout << "Target dE/dt   : " << std::setprecision(10) << P_target << "\n";
    std::cout << "l_max          = " << l_max        << "\n"
              << "N              = " << N            << "\n"
              << "seeds_used     = " << seeds.size() << "\n"
              << "best_iters     = " << best_iterations << "\n"
              << "best_du        = " << best_du << "\n"
              << "|F|_max        = " << std::setprecision(10) << lambda_max   << "\n"
              << "F_x            = " << force_x << "\n"
              << "F_y            = " << force_y << "\n"
              << "F_z            = " << force_z << "\n"
              << "||c||²         = " << actual_norm_sq << "\n"
              << "dE/dt          = " << actual_power   << "\n"
              << "\n--- Upper Bounds (Spectral Methods) ---\n"
              << "Bound (Easy)   = " << spectral_bound_easy << "  [√(||M_x||_2²+||M_y||_2²+||M_z||_2²)]\n"
              << "Bound (Tight)  = " << spectral_bound_tight << "  [√(λ_max(M_x²+M_y²+M_z²))]\n"
              << "Gap % (Easy)   = " << (100.0 * (spectral_bound_easy - lambda_max) / lambda_max) << "%\n"
              << "Gap % (Tight)  = " << (100.0 * (spectral_bound_tight - lambda_max) / lambda_max) << "%\n\n";

    if (best_iterations >= max_iter - 1 || best_du > 1e-8) {
        std::cout << "Suggestion     : Convergence is modest. Try more seeds or higher max_iter/tighter tol.\n\n";
    }

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

    return {lambda_max, force_x, force_y, force_z, actual_norm_sq, actual_power,
            static_cast<int>(seeds.size()), best_iterations, best_du, 
            spectral_bound_easy, spectral_bound_tight,
            c_opt, idx_map, N};
}

// ============================================================================
// save_coefs  –  writes CSV with header row indicating the constraint used
// ============================================================================
void save_coefs(const Result& res, int l_max, ConstraintType ct,
                double P_target, std::string folder) {
    mkdir(folder.c_str(), 0755);   // no-op if already exists

    int P = (int)P_target;

    std::string suffix = (ct == ConstraintType::Norm) ? "norm" : "power";
    std::string fname  = folder + "/maximized_coefs_xyz_lMax_" + std::to_string(l_max)
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
    f << "# Fx,"         << res.force_x << "\n";
    f << "# Fy,"         << res.force_y << "\n";
    f << "# Fz,"         << res.force_z << "\n";
    f << "# seeds_used," << res.seed_count << "\n";
    f << "# best_iters," << res.best_iterations << "\n";
    f << "# best_du,"    << res.best_du << "\n";
    f << "# spectral_bound_easy," << res.spectral_bound_easy << "\n";
    f << "# spectral_bound_tight," << res.spectral_bound_tight << "\n";
    f << "# opt_gap_easy,"   << (res.spectral_bound_easy - res.lambda_max) << "\n";
    f << "# opt_gap_tight,"  << (res.spectral_bound_tight - res.lambda_max) << "\n";

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

    std::cout << "=== Multipole XYZ-Force Maximization ===\n"
              << "l_max=" << l_max << "  r=" << r << "  Omega=" << Omega << "\n";

    Result res = solve_max_force(l_max, r, Omega, ct, P_target, /*verbose=*/true);
    save_coefs(res, l_max, ct, P_target, folder);
    return 0;
}
