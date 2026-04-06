/*
 * nlp_xyz_optimized.cpp
 *
 * Optimized XYZ force maximization with three improvements:
 * 1) Tight-bound informed seeding from top eigvec of Mx^2 + My^2 + Mz^2.
 * 2) Multi-level seeding (axes, diagonals, random, spectral seeds).
 * 3) Nonlinear conjugate-gradient refinement on the unit sphere.
 *
 * Designed for larger l_max by using sparse operators and iterative eigensolvers.
 *
 * Build:
 *   clang++ -std=c++17 -O3 -march=native nlp_xyz_optimized.cpp -o nlp_xyz_optimized -I /usr/local/include/eigen3
 *       
 *
 * Usage:
 *   ./nlp_xyz_optimized <l_max> [c] [P_target] [r] [Omega] [folder] [random_seeds] [fp_iters] [cg_iters] [target_hours] [max_stages]
 *
 * Notes:
 *   c=0 -> power constraint
 *   c=1 -> global norm constraint: ||c||^2 = l_max (legacy)
 *   c=2 -> per-l shell constraint: sum_m (|A_lm|^2 + |B_lm|^2) = 1 for each l
 */

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

using cdouble = std::complex<double>;
using VectorXcd = Eigen::VectorXcd;
using SpMatrix = Eigen::SparseMatrix<cdouble, Eigen::RowMajor>;
using Triplet = Eigen::Triplet<cdouble>;

const double PI = std::acos(-1.0);

enum class ConstraintType { NormGlobal, Power, NormPerL };

struct Key {
    int l;
    int m;
    int t; // 0=A, 1=B
    bool operator<(const Key& o) const {
        if (l != o.l) return l < o.l;
        if (m != o.m) return m < o.m;
        return t < o.t;
    }
};

using IndexMap = std::map<Key, int>;

struct Result {
    double lambda_max;
    double fx;
    double fy;
    double fz;
    double norm_sq;
    double power;
    int seed_count;
    int best_fp_iters;
    int best_cg_iters;
    double spectral_bound_easy;
    double spectral_bound_tight;
    double spectral_bound_tight_est;
    int tight_bound_is_upper;
    std::vector<cdouble> c_opt;
    IndexMap idx_map;
    int N;
};

struct Candidate {
    VectorXcd c;
    double mag;
    double fx;
    double fy;
    double fz;
    int fp_iters;
    int cg_iters;
};

IndexMap build_index_map(int l_max, int& N) {
    IndexMap idx;
    int i = 0;
    for (int l = 2; l <= l_max; ++l) {
        for (int m = -l; m <= l; ++m) {
            idx[{l, m, 0}] = i++;
            idx[{l, m, 1}] = i++;
        }
    }
    N = i;
    return idx;
}

using Shells = std::vector<std::vector<int>>;

Shells build_shells(int l_max, const IndexMap& idx_map) {
    Shells shells;
    shells.reserve(static_cast<size_t>(std::max(0, l_max - 1)));
    for (int l = 2; l <= l_max; ++l) {
        std::vector<int> shell;
        shell.reserve(static_cast<size_t>(2 * (2 * l + 1)));
        for (int m = -l; m <= l; ++m) {
            shell.push_back(idx_map.at({l, m, 0}));
            shell.push_back(idx_map.at({l, m, 1}));
        }
        shells.push_back(std::move(shell));
    }
    return shells;
}

void project_global_unit(VectorXcd& v) {
    const double n = v.norm();
    if (n < 1e-16) {
        v = VectorXcd::Ones(v.size());
        v.normalize();
        return;
    }
    v /= n;
}

void project_per_l_shells(VectorXcd& v, const Shells& shells) {
    for (const auto& shell : shells) {
        double n2 = 0.0;
        for (int idx : shell) n2 += std::norm(v(idx));
        const double n = std::sqrt(n2);
        if (n < 1e-16) {
            for (int idx : shell) v(idx) = cdouble(0.0, 0.0);
            if (!shell.empty()) v(shell.front()) = cdouble(1.0, 0.0);
        } else {
            for (int idx : shell) v(idx) /= n;
        }
    }
}

void project_constraint(VectorXcd& v, ConstraintType ct, const Shells& shells) {
    if (ct == ConstraintType::NormPerL) {
        project_per_l_shells(v, shells);
    } else {
        project_global_unit(v);
    }
}

void project_tangent(VectorXcd& g, const VectorXcd& x, ConstraintType ct, const Shells& shells) {
    if (ct == ConstraintType::NormPerL) {
        for (const auto& shell : shells) {
            cdouble proj = cdouble(0.0, 0.0);
            for (int idx : shell) proj += std::conj(x(idx)) * g(idx);
            for (int idx : shell) g(idx) -= proj * x(idx);
        }
    } else {
        const cdouble proj = x.dot(g);
        g -= proj * x;
    }
}

double constraint_norm_sq_target(ConstraintType ct,
                                 int l_max,
                                 double r,
                                 double Omega,
                                 double P_target,
                                 const Shells& shells) {
    if (ct == ConstraintType::Power) {
        return 4.0 * PI * Omega * Omega * P_target / (r * r);
    }
    if (ct == ConstraintType::NormPerL) {
        return static_cast<double>(shells.size());
    }
    return static_cast<double>(l_max);
}

static inline void add_pair(std::vector<Triplet>& t, int i, int j, cdouble v) {
    t.emplace_back(i, j, v);
    if (i != j) t.emplace_back(j, i, std::conj(v));
}

SpMatrix build_M_z(int l_max, double r, double Omega, const IndexMap& idx_map, int N) {
    std::vector<Triplet> t;
    t.reserve(static_cast<size_t>(16) * static_cast<size_t>(N));

    const double pf = 8.0 * r * r / (Omega * Omega);
    for (int l = 2; l <= l_max; ++l) {
        for (int m = -l; m <= l; ++m) {
            const int iA = idx_map.at({l, m, 0});
            const int iB = idx_map.at({l, m, 1});

            if (l + 1 <= l_max) {
                const double a = 1.0 / (32.0 * PI * (l + 1)) *
                                 std::sqrt((2.0 * (l - 1) * (l + 3)) /
                                           ((2.0 * l + 1) * (2.0 * l + 3)));
                const double alpha = a * std::sqrt(2.0 * (l - m + 1) * (l + m + 1)) * pf;
                const int jA = idx_map.at({l + 1, m, 0});
                const int jB = idx_map.at({l + 1, m, 1});
                add_pair(t, iA, jA, cdouble(alpha * 0.5, 0.0));
                add_pair(t, iB, jB, cdouble(alpha * 0.5, 0.0));
            }

            const cdouble beta = cdouble(0.0, -1.0) * static_cast<double>(m)
                               / (8.0 * PI * l * (l + 1)) * pf;
            add_pair(t, iA, iB, beta * 0.5);
        }
    }

    SpMatrix M(N, N);
    M.setFromTriplets(t.begin(), t.end(), [](const cdouble& a, const cdouble& b) { return a + b; });
    M.makeCompressed();
    return M;
}

SpMatrix build_M_x(int l_max, double r, double Omega, const IndexMap& idx_map, int N) {
    std::vector<Triplet> t;
    t.reserve(static_cast<size_t>(24) * static_cast<size_t>(N));

    const double pf = 8.0 * r * r / (Omega * Omega);
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    for (int l = 2; l <= l_max; ++l) {
        const double b_pf = pf / (4.0 * PI * l * (l + 1));
        for (int m = -l; m <= l; ++m) {
            const int iA = idx_map.at({l, m, 0});
            const int iB = idx_map.at({l, m, 1});

            if (l + 1 <= l_max) {
                const double a = 1.0 / (32.0 * PI * (l + 1)) *
                                 std::sqrt((2.0 * (l - 1) * (l + 3)) /
                                           ((2.0 * l + 1) * (2.0 * l + 3)));

                {
                    const double K = a * pf * inv_sqrt2 *
                                     std::sqrt(static_cast<double>((l - m + 1) * (l - m + 2)));
                    const int jA = idx_map.at({l + 1, m - 1, 0});
                    const int jB = idx_map.at({l + 1, m - 1, 1});
                    add_pair(t, iA, jA, cdouble(K * 0.5, 0.0));
                    add_pair(t, iB, jB, cdouble(K * 0.5, 0.0));
                }

                {
                    const double K = -a * pf * inv_sqrt2 *
                                     std::sqrt(static_cast<double>((l + m + 1) * (l + m + 2)));
                    const int jA = idx_map.at({l + 1, m + 1, 0});
                    const int jB = idx_map.at({l + 1, m + 1, 1});
                    add_pair(t, iA, jA, cdouble(K * 0.5, 0.0));
                    add_pair(t, iB, jB, cdouble(K * 0.5, 0.0));
                }
            }

            if (m > -l) {
                const double mag = b_pf * std::sqrt(0.5 * (l + m) * (l - m + 1));
                const cdouble K = cdouble(0.0, -1.0) * mag;
                const int jB = idx_map.at({l, m - 1, 1});
                add_pair(t, iA, jB, K * 0.5);
            }

            if (m < l) {
                const double mag = b_pf * std::sqrt(0.5 * (l - m) * (l + m + 1));
                const cdouble K = cdouble(0.0, -1.0) * mag;
                const int jB = idx_map.at({l, m + 1, 1});
                add_pair(t, iA, jB, K * 0.5);
            }
        }
    }

    SpMatrix M(N, N);
    M.setFromTriplets(t.begin(), t.end(), [](const cdouble& a, const cdouble& b) { return a + b; });
    M.makeCompressed();
    return M;
}

SpMatrix build_M_y(int l_max, double r, double Omega, const IndexMap& idx_map, int N) {
    std::vector<Triplet> t;
    t.reserve(static_cast<size_t>(24) * static_cast<size_t>(N));

    const double pf = 8.0 * r * r / (Omega * Omega);
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    for (int l = 2; l <= l_max; ++l) {
        const double b_pf = pf / (4.0 * PI * l * (l + 1));
        for (int m = -l; m <= l; ++m) {
            const int iA = idx_map.at({l, m, 0});
            const int iB = idx_map.at({l, m, 1});

            if (l + 1 <= l_max) {
                const double a = 1.0 / (32.0 * PI * (l + 1)) *
                                 std::sqrt((2.0 * (l - 1) * (l + 3)) /
                                           ((2.0 * l + 1) * (2.0 * l + 3)));

                {
                    const double K_mag = a * pf * inv_sqrt2 *
                                         std::sqrt(static_cast<double>((l - m + 1) * (l - m + 2)));
                    const cdouble K = cdouble(0.0, -1.0) * K_mag;
                    const int jA = idx_map.at({l + 1, m - 1, 0});
                    const int jB = idx_map.at({l + 1, m - 1, 1});
                    add_pair(t, iA, jA, K * 0.5);
                    add_pair(t, iB, jB, K * 0.5);
                }

                {
                    const double K_mag = a * pf * inv_sqrt2 *
                                         std::sqrt(static_cast<double>((l + m + 1) * (l + m + 2)));
                    const cdouble K = cdouble(0.0, -1.0) * K_mag;
                    const int jA = idx_map.at({l + 1, m + 1, 0});
                    const int jB = idx_map.at({l + 1, m + 1, 1});
                    add_pair(t, iA, jA, K * 0.5);
                    add_pair(t, iB, jB, K * 0.5);
                }
            }

            if (m > -l) {
                const double K = b_pf * std::sqrt(0.5 * (l + m) * (l - m + 1));
                const int jB = idx_map.at({l, m - 1, 1});
                add_pair(t, iA, jB, cdouble(-K * 0.5, 0.0));
            }

            if (m < l) {
                const double K = b_pf * std::sqrt(0.5 * (l - m) * (l + m + 1));
                const int jB = idx_map.at({l, m + 1, 1});
                add_pair(t, iA, jB, cdouble(K * 0.5, 0.0));
            }
        }
    }

    SpMatrix M(N, N);
    M.setFromTriplets(t.begin(), t.end(), [](const cdouble& a, const cdouble& b) { return a + b; });
    M.makeCompressed();
    return M;
}

static inline double norm2(const VectorXcd& v) {
    return v.squaredNorm();
}

static inline double qreal(const VectorXcd& c, const SpMatrix& M, VectorXcd& tmp) {
    tmp.noalias() = M * c;
    return c.dot(tmp).real();
}

struct ForceVals {
    double fx;
    double fy;
    double fz;
    double mag;
};

ForceVals compute_force(const VectorXcd& c,
                        const SpMatrix& Mx,
                        const SpMatrix& My,
                        const SpMatrix& Mz,
                        VectorXcd& tmp) {
    const double fx = qreal(c, Mx, tmp);
    const double fy = qreal(c, My, tmp);
    const double fz = qreal(c, Mz, tmp);
    const double mag = std::sqrt(std::max(0.0, fx * fx + fy * fy + fz * fz));
    return {fx, fy, fz, mag};
}

double spectral_norm_estimate(const SpMatrix& M, int iters = 80, double tol = 1e-10) {
    const int N = M.rows();
    VectorXcd v = VectorXcd::Random(N);
    v.normalize();
    VectorXcd w(N);

    double prev = 0.0;
    for (int i = 0; i < iters; ++i) {
        w.noalias() = M * v;
        const double nw = w.norm();
        if (nw < 1e-16) return 0.0;
        v = w / nw;
        w.noalias() = M * v;
        const double ev = std::abs(v.dot(w).real());
        if (std::abs(ev - prev) < tol * (1.0 + std::abs(ev))) return ev;
        prev = ev;
    }
    return prev;
}

double matrix_one_norm(const SpMatrix& M) {
    std::vector<double> col_sums(static_cast<size_t>(M.cols()), 0.0);
    for (int k = 0; k < M.outerSize(); ++k) {
        for (SpMatrix::InnerIterator it(M, k); it; ++it) {
            col_sums[static_cast<size_t>(it.col())] += std::abs(it.value());
        }
    }
    double n1 = 0.0;
    for (double s : col_sums) n1 = std::max(n1, s);
    return n1;
}

double matrix_inf_norm(const SpMatrix& M) {
    std::vector<double> row_sums(static_cast<size_t>(M.rows()), 0.0);
    for (int k = 0; k < M.outerSize(); ++k) {
        for (SpMatrix::InnerIterator it(M, k); it; ++it) {
            row_sums[static_cast<size_t>(it.row())] += std::abs(it.value());
        }
    }
    double ni = 0.0;
    for (double s : row_sums) ni = std::max(ni, s);
    return ni;
}

double two_norm_upper_bound(const SpMatrix& M) {
    const double n1 = matrix_one_norm(M);
    const double ni = matrix_inf_norm(M);
    return std::sqrt(n1 * ni);
}

VectorXcd top_eigvec_shifted_direction(const std::array<double, 3>& u,
                                       const SpMatrix& Mx,
                                       const SpMatrix& My,
                                       const SpMatrix& Mz,
                                       ConstraintType ct,
                                       const Shells& shells,
                                       double shift,
                                       int iters,
                                       double tol,
                                       const VectorXcd* init = nullptr) {
    const int N = Mx.rows();
    VectorXcd v = (init && init->size() == N) ? *init : VectorXcd::Random(N);
    project_constraint(v, ct, shells);

    VectorXcd t1(N), t2(N), t3(N), w(N);

    double prev = 0.0;
    for (int i = 0; i < iters; ++i) {
        t1.noalias() = Mx * v;
        t2.noalias() = My * v;
        t3.noalias() = Mz * v;
        w.noalias() = u[0] * t1 + u[1] * t2 + u[2] * t3 + shift * v;

        v = w;
        project_constraint(v, ct, shells);

        t1.noalias() = Mx * v;
        t2.noalias() = My * v;
        t3.noalias() = Mz * v;
        const double rq = (v.dot(u[0] * t1 + u[1] * t2 + u[2] * t3)).real();
        if (std::abs(rq - prev) < tol * (1.0 + std::abs(rq))) break;
        prev = rq;
    }
    return v;
}

VectorXcd top_eigvec_combined(const SpMatrix& Mx,
                              const SpMatrix& My,
                              const SpMatrix& Mz,
                              ConstraintType ct,
                              const Shells& shells,
                              int iters,
                              double tol) {
    const int N = Mx.rows();
    VectorXcd v = VectorXcd::Random(N);
    project_constraint(v, ct, shells);

    VectorXcd t1(N), t2(N), t3(N), w(N);
    double prev = 0.0;
    for (int i = 0; i < iters; ++i) {
        t1.noalias() = Mx * v;
        t2.noalias() = My * v;
        t3.noalias() = Mz * v;

        w.noalias() = Mx * t1;
        w += My * t2;
        w += Mz * t3;

        v = w;
        project_constraint(v, ct, shells);

        t1.noalias() = Mx * v;
        t2.noalias() = My * v;
        t3.noalias() = Mz * v;
        VectorXcd y = Mx * t1 + My * t2 + Mz * t3;
        const double rq = v.dot(y).real();
        if (std::abs(rq - prev) < tol * (1.0 + std::abs(rq))) break;
        prev = rq;
    }
    return v;
}

static inline std::array<double, 3> direction_from_force(const ForceVals& f) {
    if (f.mag < 1e-18) return {1.0, 0.0, 0.0};
    return {f.fx / f.mag, f.fy / f.mag, f.fz / f.mag};
}

void add_direction_seed(std::vector<std::array<double, 3>>& seeds, const std::array<double, 3>& u) {
    const double n = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    if (n < 1e-14) return;
    std::array<double, 3> v = {u[0] / n, u[1] / n, u[2] / n};

    for (const auto& s : seeds) {
        const double d = std::abs(s[0] * v[0] + s[1] * v[1] + s[2] * v[2]);
        if (d > 1.0 - 1e-8) return;
    }
    seeds.push_back(v);
}

void keep_top_k(std::vector<Candidate>& top, Candidate cand, size_t k) {
    top.push_back(std::move(cand));
    std::sort(top.begin(), top.end(), [](const Candidate& a, const Candidate& b) {
        return a.mag > b.mag;
    });
    if (top.size() > k) top.resize(k);
}

Candidate refine_with_ncg(const Candidate& init,
                          const SpMatrix& Mx,
                          const SpMatrix& My,
                          const SpMatrix& Mz,
                          ConstraintType ct,
                          const Shells& shells,
                          int max_iters,
                          double grad_tol) {
    Candidate cur = init;
    VectorXcd c = cur.c;
    project_constraint(c, ct, shells);

    VectorXcd tmp(c.size()), tx(c.size()), ty(c.size()), tz(c.size());

    auto grad_eval = [&](const VectorXcd& x, ForceVals& f, VectorXcd& g_out) {
        f = compute_force(x, Mx, My, Mz, tmp);
        tx.noalias() = Mx * x;
        ty.noalias() = My * x;
        tz.noalias() = Mz * x;

        const double denom = std::max(f.mag, 1e-16);
        g_out.noalias() = (f.fx * tx + f.fy * ty + f.fz * tz) / denom;

        project_tangent(g_out, x, ct, shells);
    };

    ForceVals fcur;
    VectorXcd g(c.size()), g_prev(c.size()), d(c.size());
    grad_eval(c, fcur, g);
    d = g;

    double g2_prev = std::max(1e-30, g.squaredNorm());

    int used = 0;
    for (int it = 0; it < max_iters; ++it) {
        const double gn = std::sqrt(g.squaredNorm());
        if (gn < grad_tol) {
            used = it;
            break;
        }

        double alpha = 0.5;
        bool accepted = false;
        ForceVals fnew = fcur;
        VectorXcd cnew = c;

        for (int ls = 0; ls < 20; ++ls) {
            cnew = c + alpha * d;
            project_constraint(cnew, ct, shells);
            fnew = compute_force(cnew, Mx, My, Mz, tmp);

            if (fnew.mag > fcur.mag + 1e-6 * alpha * gn * gn) {
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }

        if (!accepted) {
            d = g;
            used = it;
            continue;
        }

        c = cnew;
        fcur = fnew;

        g_prev = g;
        grad_eval(c, fcur, g);

        const VectorXcd y = g - g_prev;
        const double num = std::max(0.0, y.dot(g).real());
        const double den = std::max(1e-30, g2_prev);
        const double beta = std::max(0.0, num / den);

        d = g + beta * d;
        project_tangent(d, c, ct, shells);

        g2_prev = std::max(1e-30, g.squaredNorm());
        used = it + 1;
    }

    cur.c = c;
    cur.fp_iters = init.fp_iters;
    cur.cg_iters = used;
    cur.fx = fcur.fx;
    cur.fy = fcur.fy;
    cur.fz = fcur.fz;
    cur.mag = fcur.mag;
    return cur;
}

Result solve_max_force_optimized(int l_max,
                                 double r,
                                 double Omega,
                                 ConstraintType ct,
                                 double P_target,
                                 int random_seed_count,
                                 int fp_max_iters,
                                 int cg_max_iters,
                                 bool verbose) {
    int N = 0;
    IndexMap idx_map = build_index_map(l_max, N);
    Shells shells = build_shells(l_max, idx_map);

    SpMatrix Mz = build_M_z(l_max, r, Omega, idx_map, N);
    SpMatrix Mx = build_M_x(l_max, r, Omega, idx_map, N);
    SpMatrix My = build_M_y(l_max, r, Omega, idx_map, N);

    const double norm_sq_target = constraint_norm_sq_target(ct, l_max, r, Omega, P_target, shells);

    // Rigorous per-matrix upper bounds: ||M||_2 <= sqrt(||M||_1 * ||M||_inf)
    const double sx_up = two_norm_upper_bound(Mx);
    const double sy_up = two_norm_upper_bound(My);
    const double sz_up = two_norm_upper_bound(Mz);
    const double easy_factor = std::sqrt(sx_up * sx_up + sy_up * sy_up + sz_up * sz_up);

    // For seeding and diagnostics, keep a power-method estimate of top eigvec of Mx^2+My^2+Mz^2.
    VectorXcd comb_top = top_eigvec_combined(Mx, My, Mz, ct, shells, 120, 1e-10);
    VectorXcd tmp(N), tx(N), ty(N), tz(N);
    tx.noalias() = Mx * comb_top;
    ty.noalias() = My * comb_top;
    tz.noalias() = Mz * comb_top;
    VectorXcd y = Mx * tx + My * ty + Mz * tz;
    const double tight_factor_est = std::sqrt(std::max(0.0, comb_top.dot(y).real()));

    // Rigorous tight upper bound via A = Mx^2 + My^2 + Mz^2 and lambda_max(A) <= ||A||_inf.
    SpMatrix A = (Mx * Mx + My * My + Mz * Mz).pruned(0.0);
    const double A_inf = matrix_inf_norm(A);
    const double tight_factor_upper = std::sqrt(std::max(0.0, A_inf));

    const double spectral_bound_easy = norm_sq_target * easy_factor;
    const double spectral_bound_tight_est = norm_sq_target * tight_factor_est;

    // For moderate N, compute exact tight bound by dense Hermitian eigensolve of A.
    // For large N, fall back to rigorous upper bound from ||A||_inf.
    constexpr int exact_tight_bound_max_n = 800;
    double spectral_bound_tight = norm_sq_target * tight_factor_upper;
    int tight_bound_is_upper = 1;
    if (N <= exact_tight_bound_max_n) {
        Eigen::MatrixXcd A_dense = Eigen::MatrixXcd(A);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigA(A_dense, Eigen::EigenvaluesOnly);
        if (eigA.info() == Eigen::Success) {
            const double lam_max = std::max(0.0, eigA.eigenvalues().maxCoeff());
            spectral_bound_tight = norm_sq_target * std::sqrt(lam_max);
            tight_bound_is_upper = 0;
        }
    }

    std::vector<std::array<double, 3>> seeds;
    seeds.reserve(static_cast<size_t>(32 + random_seed_count));

    const std::array<std::array<double, 3>, 13> base = {{
        {{1.0, 0.0, 0.0}}, {{-1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}}, {{0.0, -1.0, 0.0}},
        {{0.0, 0.0, 1.0}}, {{0.0, 0.0, -1.0}},
        {{1.0, 1.0, 1.0}}, {{1.0, -1.0, 1.0}},
        {{-1.0, 1.0, 1.0}}, {{1.0, 1.0, -1.0}},
        {{-1.0, -1.0, 1.0}}, {{-1.0, 1.0, -1.0}},
        {{1.0, -1.0, -1.0}}
    }};

    for (const auto& b : base) add_direction_seed(seeds, b);

    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int i = 0; i < random_seed_count; ++i) {
        add_direction_seed(seeds, {nd(rng), nd(rng), nd(rng)});
    }

    auto add_seed_from_c = [&](const VectorXcd& cseed) {
        VectorXcd v = cseed;
        if (v.norm() < 1e-16) return;
        v.normalize();
        ForceVals f = compute_force(v, Mx, My, Mz, tmp);
        add_direction_seed(seeds, direction_from_force(f));
    };

    add_seed_from_c(comb_top);
    add_seed_from_c(top_eigvec_shifted_direction({1.0, 0.0, 0.0}, Mx, My, Mz, ct, shells, easy_factor + 1e-6, 80, 1e-8));
    add_seed_from_c(top_eigvec_shifted_direction({0.0, 1.0, 0.0}, Mx, My, Mz, ct, shells, easy_factor + 1e-6, 80, 1e-8));
    add_seed_from_c(top_eigvec_shifted_direction({0.0, 0.0, 1.0}, Mx, My, Mz, ct, shells, easy_factor + 1e-6, 80, 1e-8));

    const size_t top_k = 5;
    std::vector<Candidate> top_candidates;
    top_candidates.reserve(top_k);

    int best_fp_iters = 0;
    double best_du = 1e300;

    for (const auto& seed : seeds) {
        std::array<double, 3> u = seed;
        VectorXcd c = top_eigvec_shifted_direction(u, Mx, My, Mz, ct, shells, easy_factor + 1e-6, 80, 1e-8);

        double fx = 0.0, fy = 0.0, fz = 0.0;
        int fp_used = 0;
        double du_last = 1e300;

        for (int it = 0; it < fp_max_iters; ++it) {
            c = top_eigvec_shifted_direction(u, Mx, My, Mz, ct, shells, easy_factor + 1e-6, 50, 1e-8, &c);

            tx.noalias() = Mx * c;
            ty.noalias() = My * c;
            tz.noalias() = Mz * c;
            fx = c.dot(tx).real();
            fy = c.dot(ty).real();
            fz = c.dot(tz).real();

            const double mag = std::sqrt(std::max(0.0, fx * fx + fy * fy + fz * fz));
            if (mag < 1e-16) break;

            const std::array<double, 3> u_new = {fx / mag, fy / mag, fz / mag};
            const double du = std::sqrt((u_new[0] - u[0]) * (u_new[0] - u[0]) +
                                        (u_new[1] - u[1]) * (u_new[1] - u[1]) +
                                        (u_new[2] - u[2]) * (u_new[2] - u[2]));
            du_last = du;
            fp_used = it + 1;
            u = u_new;
            if (du < 1e-10) break;
        }

        const double mag = std::sqrt(std::max(0.0, fx * fx + fy * fy + fz * fz));
        Candidate cand{c, mag, fx, fy, fz, fp_used, 0};
        keep_top_k(top_candidates, std::move(cand), top_k);

        if (mag >= top_candidates.front().mag) {
            best_fp_iters = fp_used;
            best_du = du_last;
        }
    }

    Candidate best = top_candidates.front();
    for (const auto& cand : top_candidates) {
        Candidate refined = refine_with_ncg(cand, Mx, My, Mz, ct, shells, cg_max_iters, 1e-10);
        if (refined.mag > best.mag) best = refined;
    }

    const bool is_unit_sphere_mode = (ct != ConstraintType::NormPerL);
    const double scale = is_unit_sphere_mode ? std::sqrt(norm_sq_target) : 1.0;
    VectorXcd c_scaled = is_unit_sphere_mode ? (scale * best.c) : best.c;

    std::vector<cdouble> c_opt(N);
    for (int i = 0; i < N; ++i) c_opt[i] = c_scaled(i);

    const double lambda_max = is_unit_sphere_mode ? (best.mag * norm_sq_target) : best.mag;
    const double fx = is_unit_sphere_mode ? (best.fx * norm_sq_target) : best.fx;
    const double fy = is_unit_sphere_mode ? (best.fy * norm_sq_target) : best.fy;
    const double fz = is_unit_sphere_mode ? (best.fz * norm_sq_target) : best.fz;

    const double actual_norm_sq = norm_sq_target;
    const double actual_power = (r * r) / (4.0 * PI * Omega * Omega) * actual_norm_sq;

    if (verbose) {
        std::cout << "\n=== XYZ Force Maximization (Optimized) ===\n";
        std::cout << "Constraint     : ";
        if (ct == ConstraintType::Power) {
            std::cout << "dE/dt = P\n";
        } else if (ct == ConstraintType::NormPerL) {
            std::cout << "sum_m(|A_lm|^2+|B_lm|^2)=1 for each l\n";
        } else {
            std::cout << "||c||^2 = l_max\n";
            std::cout << "Note           : using global norm constraint on c (not per-l shell normalization).\n";
        }
        std::cout << "l_max          = " << l_max << "\n";
        std::cout << "N              = " << N << "\n";
        std::cout << "seeds_used     = " << seeds.size() << "\n";
        std::cout << "best_fp_iters  = " << best_fp_iters << "\n";
        std::cout << "best_cg_iters  = " << best.cg_iters << "\n";
        std::cout << "best_du        = " << best_du << "\n";
        std::cout << "|F|_max        = " << std::setprecision(10) << lambda_max << "\n";
        std::cout << "F_x            = " << fx << "\n";
        std::cout << "F_y            = " << fy << "\n";
        std::cout << "F_z            = " << fz << "\n";
        std::cout << "||c||^2        = " << actual_norm_sq << "\n";
        std::cout << "dE/dt          = " << actual_power << "\n\n";

        std::cout << "--- Spectral Upper Bounds ---\n";
        std::cout << "Bound (Easy)   = " << spectral_bound_easy << "\n";
        std::cout << "Bound (Tight)  = " << spectral_bound_tight;
        if (tight_bound_is_upper) {
            std::cout << "  [upper via ||A||_inf]";
        } else {
            std::cout << "  [exact lambda_max(A)]";
        }
        std::cout << "\n";
        std::cout << "Tight est (PM) = " << spectral_bound_tight_est << "  [not guaranteed upper]\n";
        std::cout << "Gap % (Easy)   = " << (100.0 * (spectral_bound_easy - lambda_max) / lambda_max) << "%\n";
        std::cout << "Gap % (Tight)  = " << (100.0 * (spectral_bound_tight - lambda_max) / lambda_max) << "%\n\n";
    }

    if (verbose) {
        std::cout << std::setw(18) << "(l,m,type)"
                  << std::setw(16) << "Re(c)"
                  << std::setw(16) << "Im(c)"
                  << std::setw(14) << "|c|" << "\n"
                  << std::string(64, '-') << "\n";
        for (const auto& kv : idx_map) {
            const Key& key = kv.first;
            const int idx = kv.second;
            const cdouble val = c_opt[idx];
            if (std::abs(val) < 1e-8) continue;
            std::cout << "  (" << key.l << "," << std::showpos << key.m << std::noshowpos
                      << "," << (key.t == 0 ? 'A' : 'B') << ")"
                      << std::setw(16) << std::fixed << std::setprecision(6) << val.real()
                      << std::setw(16) << val.imag()
                      << std::setw(14) << std::abs(val) << "\n";
        }
    }

    return {lambda_max,
            fx,
            fy,
            fz,
            actual_norm_sq,
            actual_power,
            static_cast<int>(seeds.size()),
            best_fp_iters,
            best.cg_iters,
            spectral_bound_easy,
            spectral_bound_tight,
            spectral_bound_tight_est,
            tight_bound_is_upper,
            c_opt,
            idx_map,
            N};
}

Result solve_with_time_budget(int l_max,
                              double r,
                              double Omega,
                              ConstraintType ct,
                              double P_target,
                              int base_random_seeds,
                              int base_fp_iters,
                              int base_cg_iters,
                              double target_hours,
                              int max_stages,
                              bool verbose_final) {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    const double budget_s = std::max(0.0, target_hours) * 3600.0;

    Result best{};
    bool have_best = false;

    for (int stage = 0; stage < max_stages; ++stage) {
        const double elapsed_s = std::chrono::duration<double>(Clock::now() - start).count();
        if (budget_s > 0.0 && elapsed_s >= budget_s) break;

        const int random_seeds = base_random_seeds + 16 * stage;
        const int fp_iters = base_fp_iters + 20 * stage;
        const int cg_iters = base_cg_iters + 30 * stage;

        std::cout << "[schedule] stage=" << (stage + 1)
                  << " seeds=" << random_seeds
                  << " fp_iters=" << fp_iters
                  << " cg_iters=" << cg_iters;
        if (budget_s > 0.0) {
            std::cout << " remaining_s=" << std::max(0.0, budget_s - elapsed_s);
        }
        std::cout << "\n";

        const auto stage_start = Clock::now();
        Result cur = solve_max_force_optimized(l_max, r, Omega, ct, P_target,
                                               random_seeds, fp_iters, cg_iters,
                                               /*verbose=*/false);
        const double stage_s = std::chrono::duration<double>(Clock::now() - stage_start).count();

        if (!have_best || cur.lambda_max > best.lambda_max) {
            best = std::move(cur);
            have_best = true;
            std::cout << "[schedule] new_best=" << std::setprecision(10) << best.lambda_max << "\n";
        } else {
            std::cout << "[schedule] stage_result=" << std::setprecision(10) << cur.lambda_max
                      << " best=" << best.lambda_max << "\n";
        }
        std::cout << "[schedule] stage_time_s=" << stage_s << "\n";

        if (budget_s > 0.0) {
            const double now_elapsed_s = std::chrono::duration<double>(Clock::now() - start).count();
            const double remaining_s = budget_s - now_elapsed_s;
            if (remaining_s <= 0.0) break;
            if (stage_s > 0.0 && remaining_s < 0.75 * stage_s) break;
        }
    }

    if (!have_best) {
        throw std::runtime_error("Auto-scheduler did not run any stage; increase target_hours or max_stages.");
    }

    if (verbose_final) {
        const double elapsed_s = std::chrono::duration<double>(Clock::now() - start).count();
        std::cout << "\n=== Auto-Scheduler Summary ===\n"
                  << "target_hours    = " << target_hours << "\n"
                  << "elapsed_hours   = " << (elapsed_s / 3600.0) << "\n"
                  << "best |F|_max    = " << std::setprecision(10) << best.lambda_max << "\n"
                  << "best seeds_used = " << best.seed_count << "\n"
                  << "best fp_iters   = " << best.best_fp_iters << "\n"
                  << "best cg_iters   = " << best.best_cg_iters << "\n\n";
    }

    return best;
}

void save_coefs(const Result& res, int l_max, ConstraintType ct, double P_target, const std::string& folder) {
    mkdir(folder.c_str(), 0755);

    const int P = static_cast<int>(P_target);
    const std::string suffix =
        (ct == ConstraintType::Power) ? "power" :
        (ct == ConstraintType::NormPerL) ? "norm_per_l" : "norm";
    const std::string fname = folder + "/maximized_coefs_xyz_lMax_" + std::to_string(l_max)
                            + "_Power_" + std::to_string(P) + "_" + suffix + ".csv";

    std::ofstream f(fname);
    if (!f) throw std::runtime_error("Cannot open " + fname);

    std::vector<double> A_re, A_im, B_re, B_im;
    std::vector<int> Ls, Ms;
    constexpr double coeff_save_threshold = 1e-13;

    for (const auto& kv : res.idx_map) {
        const Key& key = kv.first;
        if (key.t != 0) continue;
        const cdouble A = res.c_opt[res.idx_map.at({key.l, key.m, 0})];
        const cdouble B = res.c_opt[res.idx_map.at({key.l, key.m, 1})];
        if (std::abs(A) + std::abs(B) > coeff_save_threshold) {
            A_re.push_back(A.real());
            A_im.push_back(A.imag());
            B_re.push_back(B.real());
            B_im.push_back(B.imag());
            Ls.push_back(key.l);
            Ms.push_back(key.m);
        }
    }

    f << std::scientific << std::setprecision(16);

    auto row = [&](const std::string& lbl, const auto& v) {
        f << lbl;
        for (auto x : v) f << "," << x;
        f << "\n";
    };

    row("A_re", A_re);
    row("A_im", A_im);
    row("B_re", B_re);
    row("B_im", B_im);
    row("M", Ms);
    row("L", Ls);

        f << "# constraint,"
            << ((ct == ConstraintType::Power) ? "power" :
                    (ct == ConstraintType::NormPerL) ? "norm_per_l" : "norm")
            << "\n";
    f << "# norm_sq," << res.norm_sq << "\n";
    f << "# dEdt," << res.power << "\n";
    f << "# lambda_max," << res.lambda_max << "\n";
    f << "# Fx," << res.fx << "\n";
    f << "# Fy," << res.fy << "\n";
    f << "# Fz," << res.fz << "\n";
    f << "# seeds_used," << res.seed_count << "\n";
    f << "# best_fp_iters," << res.best_fp_iters << "\n";
    f << "# best_cg_iters," << res.best_cg_iters << "\n";
    f << "# spectral_bound_easy," << res.spectral_bound_easy << "\n";
    f << "# spectral_bound_tight," << res.spectral_bound_tight << "\n";
    f << "# spectral_bound_tight_est," << res.spectral_bound_tight_est << "\n";
    f << "# spectral_bound_tight_is_upper," << res.tight_bound_is_upper << "\n";
    f << "# opt_gap_easy," << (res.spectral_bound_easy - res.lambda_max) << "\n";
    f << "# opt_gap_tight," << (res.spectral_bound_tight - res.lambda_max) << "\n";
    f << "# coeff_save_threshold," << coeff_save_threshold << "\n";

    std::cout << "Saved to " << fname << "\n";
}

int main(int argc, char* argv[]) {
    const int l_max = (argc > 1) ? std::stoi(argv[1]) : 10;
    const int c = (argc > 2) ? std::stoi(argv[2]) : 1;
    const double P_target = (argc > 3) ? std::stod(argv[3]) : 1.0;
    const double r = (argc > 4) ? std::stod(argv[4]) : 1.0;
    const double Omega = (argc > 5) ? std::stod(argv[5]) : 1.0;
    const std::string folder = (argc > 6) ? argv[6] : "csvs";

    const int random_seeds = (argc > 7) ? std::stoi(argv[7]) : 24;
    const int fp_iters = (argc > 8) ? std::stoi(argv[8]) : 80;
    const int cg_iters = (argc > 9) ? std::stoi(argv[9]) : 120;
    const double target_hours = (argc > 10) ? std::stod(argv[10]) : 0.0;
    const int max_stages = (argc > 11) ? std::stoi(argv[11]) : 1000000;

    const ConstraintType ct =
        (c == 0) ? ConstraintType::Power :
        (c == 2) ? ConstraintType::NormPerL : ConstraintType::NormGlobal;

    std::cout << "=== Multipole XYZ-Force Maximization (Optimized) ===\n";
    std::cout << "l_max=" << l_max << " r=" << r << " Omega=" << Omega
              << " random_seeds=" << random_seeds
              << " fp_iters=" << fp_iters
              << " cg_iters=" << cg_iters;
    if (target_hours > 0.0) std::cout << " target_hours=" << target_hours;
    std::cout << "\n";

    Result res;
    if (target_hours > 0.0) {
        res = solve_with_time_budget(l_max, r, Omega, ct, P_target,
                                     random_seeds, fp_iters, cg_iters,
                                     target_hours, max_stages,
                                     true);
    } else {
        res = solve_max_force_optimized(l_max, r, Omega, ct, P_target,
                                        random_seeds, fp_iters, cg_iters,
                                        true);
    }
    save_coefs(res, l_max, ct, P_target, folder);
    return 0;
}
