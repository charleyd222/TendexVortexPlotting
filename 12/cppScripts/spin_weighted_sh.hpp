#pragma once
/**
 * spin_weighted_sh.hpp  –  Spin-weighted spherical harmonics  ₛYₗₘ(θ,φ)
 *
 * Optimized C++17 single-header.
 *
 * Key changes vs. original sylm.hpp
 * -----------------------------------
 *  1. compute_H(R)          – runs WignerH recurrence once per quaternion,
 *                             caching result in the object.
 *  2. fill_Y(spin, out)     – uses the cached H to fill Y for any spin.
 *                             Call once per spin after a single compute_H.
 *  3. Constructor takes mp_max explicitly (defaults to 2).
 *     Both s=+2 and s=-2 share mp_max=2, so one object covers both.
 *  4. sYlm_batch removed – the fused loop in main.cpp replaces it.
 *  5. Legacy compute(R, out) still works (calls compute_H + fill_Y(spin_)).
 *
 * Algorithm
 * ----------
 * Wigner H matrix via the five-step recurrence of Gumerov & Duraiswami
 * (arXiv:1403.7698), wedge storage exploiting sym A/B.
 *
 *   H(n,m',m) = H(n,m,m')       [sym A]
 *   H(n,m',m) = H(n,−m',−m)     [sym B]
 *
 *   sYlm(R) = (−1)^s √((2l+1)/4π) D^l_{m,−s}(R)
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace sylm {

using cd = std::complex<double>;

static constexpr double PI        = 3.14159265358979323846264338327950288;
static constexpr double INV_4PI   = 1.0 / (4.0 * PI);
static constexpr double SQRT3     = 1.73205080756887729352744634150587237;
static constexpr double INV_SQRT2 = 0.70710678118654752440084436210484904;

// -----------------------------------------------------------------------
//  Index helpers
// -----------------------------------------------------------------------

inline int nm_index   (int n, int m)     noexcept { return m + n * (n + 1); }
inline int nabsm_index(int n, int absm)  noexcept { return absm + (n * (n + 1)) / 2; }
inline int Yindex     (int ell, int m)   noexcept { return ell * ell + ell + m; }
inline int Ysize      (int ell_max)      noexcept { return (ell_max + 1) * (ell_max + 1); }

inline double epsilon(int m) noexcept {
    if (m <= 0) return 1.0;
    return (m & 1) ? -1.0 : 1.0;
}

// -----------------------------------------------------------------------
//  WignerH  (unchanged logic; no interface changes needed here)
// -----------------------------------------------------------------------
class WignerH {
public:
    const int n_max;
    const int mp_max;

    std::vector<double> Hwedge;
    std::vector<double> Hv;
    std::vector<double> Hextra;

private:
    std::vector<int>    _rowstart;
    std::vector<double> _a, _b, _d, _g, _h;

public:
    explicit WignerH(int n_max_, int mp_max_)
        : n_max(n_max_), mp_max(mp_max_)
    {
        _build_rowstart();
        Hwedge.assign(_hwedge_size(), 0.0);
        Hv.assign(nm_index(n_max + 1, n_max + 1) + 2, 0.0);
        Hextra.assign(n_max + 2, 0.0);
        _precompute_coeffs();
    }

    inline int hindex(int n, int mp, int m) const noexcept {
        int abs_mp = std::abs(mp), abs_m = std::abs(m);
        if (abs_mp > abs_m) { int t=mp; mp=m; m=t; t=abs_mp; abs_mp=abs_m; abs_m=t; }
        if (m < 0)          { mp=-mp; m=-m; }
        return _rowstart[n * (2 * mp_max + 1) + (mp + mp_max)] + (m - abs_mp);
    }

    void compute(cd expiβ) {
        std::fill(Hwedge.begin(), Hwedge.end(), 0.0);
        std::fill(Hv.begin(),     Hv.end(),     0.0);
        std::fill(Hextra.begin(), Hextra.end(), 0.0);
        _step_1();
        _step_2(expiβ);
        _step_3(expiβ);
        _step_4();
        _step_5();
    }

private:
    void _build_rowstart() {
        const int W = 2 * mp_max + 1;
        _rowstart.assign((n_max + 1) * W, 0);
        int flat = 0;
        for (int n = 0; n <= n_max; ++n) {
            int M = std::min(n, mp_max);
            for (int mp = -M; mp <= M; ++mp) {
                _rowstart[n * W + (mp + mp_max)] = flat;
                flat += n - std::abs(mp) + 1;
            }
        }
    }

    int _hwedge_size() const {
        int tot = 0;
        for (int n = 0; n <= n_max; ++n) {
            int M = std::min(n, mp_max);
            for (int mp = -M; mp <= M; ++mp) tot += n - std::abs(mp) + 1;
        }
        return tot;
    }

    void _precompute_coeffs() {
        const int N       = n_max + 2;
        const int nm_sz   = nm_index(N, N) + 2;
        const int nabs_sz = nabsm_index(N, N) + 2;

        _a.assign(nabs_sz, 0.0);
        _b.assign(nm_sz,   0.0);
        _d.assign(nm_sz,   0.0);
        _g.assign(nm_sz,   0.0);
        _h.assign(nm_sz,   0.0);

        for (int n = 0; n <= n_max + 1; ++n) {
            for (int absm = 0; absm <= n; ++absm) {
                int idx = nabsm_index(n, absm);
                if (idx < (int)_a.size()) {
                    double num = (double)(n+1+absm)*(n+1-absm);
                    double den = (double)(2*n+1)*(2*n+3);
                    _a[idx] = std::sqrt(num / den);
                }
            }
            for (int m = -n; m <= n; ++m) {
                int idx = nm_index(n, m);
                if (idx >= (int)_b.size()) continue;

                double nm1 = (double)(n-m-1), nm0 = (double)(n-m);
                double t1  = (double)(2*n-1),  t2  = (double)(2*n+1);
                double bv  = (nm1>=0 && nm0>=0 && t1>0) ? std::sqrt(nm1*nm0/(t1*t2)) : 0.0;
                _b[idx] = (m < 0) ? -bv : bv;

                double pm = (double)(n-m), pp = (double)(n+m+1);
                double dv = (pm>=0 && pp>=0) ? 0.5*std::sqrt(pm*pp) : 0.0;
                _d[idx] = (m < 0) ? -dv : dv;

                if (pm > 0 && pp > 0) {
                    _g[idx] = 2.0*(m+1)/std::sqrt(pm*pp);
                    double nm2=(double)(n+m+2), pm2=(double)(n-m-1);
                    _h[idx] = (nm2>=0&&pm2>=0) ? std::sqrt(nm2*pm2/(pm*pp)) : 0.0;
                }
            }
        }
    }

    void _step_1() { Hwedge[0] = 1.0; }

    void _step_2(cd expiβ) {
        const double cosβ = expiβ.real(), sinβ = expiβ.imag();
        if (n_max < 1) return;
        {
            Hwedge[hindex(1,0,1)] = SQRT3;
            Hwedge[hindex(1,0,0)] = _g[nm_index(1,0)] * cosβ * INV_SQRT2;
        }
        for (int n = 2; n <= n_max + 1; ++n) {
            double* H; int n0n;
            if (n <= n_max) { H = Hwedge.data(); n0n = hindex(n,0,n); }
            else            { H = Hextra.data(); n0n = n; }

            H[n0n] = std::sqrt(1.0 + 0.5/n) * Hwedge[hindex(n-1,0,n-1)];
            H[n0n-1] = _g[nm_index(n,n)-1] * cosβ * H[n0n];
            for (int i = 2; i < n; ++i) {
                int gi = nm_index(n,n)-i;
                H[n0n-i] = _g[gi]*cosβ*H[n0n-i+1] - _h[gi]*sinβ*sinβ*H[n0n-i+2];
            }
            double norm = 1.0 / std::sqrt(4.0*n + 2.0);
            {
                int gi = nm_index(n,n)-n;
                H[n0n-n] = (_g[gi]*cosβ*H[n0n-n+1] - _h[gi]*sinβ*sinβ*H[n0n-n+2]) * norm;
            }
            double pf = norm;
            for (int i = 1; i < n; ++i) { pf *= sinβ; H[n0n-n+i] *= pf; }
            if (n <= n_max) {
                Hv[nm_index(n,1)] = Hwedge[hindex(n,0,1)];
                Hv[nm_index(n,0)] = Hwedge[hindex(n,0,1)];
            }
        }
        {
            double pf = 1.0;
            for (int n = 1; n <= n_max; ++n) {
                pf *= sinβ;
                Hwedge[hindex(n,0,n)] *= pf / std::sqrt(4.0*n + 2.0);
            }
            { int n=n_max+1; pf*=sinβ; Hextra[n]*=pf/std::sqrt(4.0*n+2.0); }
        }
        Hv[nm_index(1,1)] = Hwedge[hindex(1,0,1)];
        Hv[nm_index(1,0)] = Hwedge[hindex(1,0,1)];
    }

    void _step_3(cd expiβ) {
        const double cosβ = expiβ.real(), sinβ = expiβ.imag();
        if (n_max < 1 || mp_max < 1) return;
        for (int n = 1; n <= n_max; ++n) {
            int i1 = hindex(n,1,1);
            double* H2; int i2;
            if (n+1 <= n_max) { H2=Hwedge.data(); i2=hindex(n+1,0,0); }
            else               { H2=Hextra.data(); i2=0; }
            int i3=nm_index(n+1,0), i4=nabsm_index(n,1);
            double inv_b5 = 1.0/_b[i3];
            for (int i = 0; i < n; ++i) {
                double b6=_b[-i+i3-2], b7=_b[i+i3], a8=_a[i+i4];
                Hwedge[i+i1] = inv_b5*(
                    0.5*(b6*(1.0-cosβ)*H2[i+i2+2] - b7*(1.0+cosβ)*H2[i+i2])
                    - a8*sinβ*H2[i+i2+1]
                );
            }
        }
    }

    void _step_4() {
        if (n_max < 1 || mp_max < 1) return;
        for (int n = 2; n <= n_max; ++n) {
            for (int mp = 1; mp < std::min(n, mp_max); ++mp) {
                int i1=hindex(n,mp+1,mp+1)-1, i2=hindex(n,mp-1,mp);
                int i3=hindex(n,mp,mp)-1,      i4=hindex(n,mp,mp+1);
                int i5=nm_index(n,mp),          i6=nm_index(n,mp-1);
                double inv_d5=1.0/_d[i5], d6=_d[i6];
                {
                    Hv[nm_index(n,mp+1)] = inv_d5*(
                        d6*Hwedge[i2] - _d[i6]*Hv[nm_index(n,mp)] + _d[i5]*Hwedge[i4]
                    );
                }
                for (int i = 1; i < n-mp; ++i)
                    Hwedge[i+i1] = inv_d5*(d6*Hwedge[i+i2] - _d[i+i6]*Hwedge[i+i3] + _d[i+i5]*Hwedge[i+i4]);
                {
                    int i=n-mp;
                    Hwedge[i+i1] = inv_d5*(d6*Hwedge[i+i2] - _d[i+i6]*Hwedge[i+i3]);
                }
            }
        }
    }

    void _step_5() {
        if (n_max < 1 || mp_max < 1) return;
        for (int n = 0; n <= n_max; ++n) {
            for (int mp = 0; mp > -std::min(n, mp_max); --mp) {
                int i1=hindex(n,mp-1,-mp+1)-1, i2=hindex(n,mp+1,-mp+1)-1;
                int i3=hindex(n,mp,-mp)-1,      i4=hindex(n,mp,-mp+1);
                int i5=nm_index(n,mp-1), i6=nm_index(n,mp);
                int i7=nm_index(n,-mp-1), i8=nm_index(n,-mp);
                double inv_d5=1.0/_d[i5], d6=_d[i6];
                {
                    double d7=_d[i7], d8=_d[i8];
                    if (mp == 0)
                        Hv[nm_index(n,mp-1)] = inv_d5*(d6*Hv[nm_index(n,mp+1)]+d7*Hv[nm_index(n,mp)]-d8*Hwedge[i4]);
                    else
                        Hv[nm_index(n,mp-1)] = inv_d5*(d6*Hwedge[i2]+d7*Hv[nm_index(n,mp)]-d8*Hwedge[i4]);
                }
                for (int i = 1; i < n+mp; ++i)
                    Hwedge[i+i1] = inv_d5*(d6*Hwedge[i+i2]+_d[i+i7]*Hwedge[i+i3]-_d[i+i8]*Hwedge[i+i4]);
                {
                    int i=n+mp;
                    Hwedge[i+i1] = inv_d5*(d6*Hwedge[i+i2]+_d[i+i7]*Hwedge[i+i3]);
                }
            }
        }
    }
};  // class WignerH

// -----------------------------------------------------------------------
//  Euler phases from unit quaternion
// -----------------------------------------------------------------------
struct EulerPhases { cd za, zg, expib; };

inline EulerPhases euler_phases(const std::array<double,4>& R) noexcept {
    const double w=R[0], x=R[1], y=R[2], z=R[3];
    cd zp_raw(w,z), zm_raw(y,x);
    double mag_zp=std::abs(zp_raw), mag_zm=std::abs(zm_raw);
    cd zp=(mag_zp>0)?(zp_raw/mag_zp):cd(1,0);
    cd zm=(mag_zm>0)?(zm_raw/mag_zm):cd(1,0);
    double cosb=mag_zp*mag_zp-mag_zm*mag_zm, sinb=2.0*mag_zp*mag_zm;
    return { zp*zm, zp*std::conj(zm), cd(cosb,sinb) };
}

inline void complex_powers(cd z, int n_max, std::vector<cd>& out) {
    out.resize(n_max+1);
    out[0]=1.0;
    for (int k=1;k<=n_max;++k) out[k]=out[k-1]*z;
}

// -----------------------------------------------------------------------
//  SphericalHarmonics
//
//  OPTIMISED INTERFACE:
//
//    SphericalHarmonics sh(lMax, /*mp_max=*/2);
//
//    // Per grid point:
//    sh.compute_H(R);          // Wigner recurrence – run ONCE per point
//    sh.fill_Y(+2, Yp, out);   // extract s=+2 into out
//    sh.fill_Y(-2, Ym, out);   // extract s=-2 into out  (no extra H work)
//
//    // Legacy path (single spin, backward-compatible):
//    sh.compute(R, out);        // calls compute_H + fill_Y(spin_)
// -----------------------------------------------------------------------
class SphericalHarmonics {
public:
    const int ell_max;
    const int mp_max;    ///< = max |spin| this object handles (typically 2)

    /// @param ell_max_  Maximum angular momentum.
    /// @param mp_max_   Maximum |spin| needed.  Defaults to 2 (for s=±2).
    explicit SphericalHarmonics(int ell_max_, int mp_max_ = 2)
        : ell_max(ell_max_), mp_max(mp_max_),
          _H(ell_max_, mp_max_),
          _spin_default(mp_max_)   // legacy: spin == mp_max
    {}

    int output_size()             const noexcept { return Ysize(ell_max); }
    int Yindex(int ell, int m)    const noexcept { return sylm::Yindex(ell, m); }

    //  Phase 1: run WignerH recurrence and cache za-powers and zg.
    //  Call once per quaternion / grid point.
    void compute_H(const std::array<double,4>& R) {
        auto ep = euler_phases(R);
        _last_zg = ep.zg;
        _H.compute(ep.expib);
        complex_powers(ep.za, ell_max, _za_pow);
    }

    //  Phase 2: fill output array for a specific spin using cached H.
    //  Can be called multiple times with different spins after one
    //  compute_H().
    //  |spin| must be <= mp_max supplied to the constructor.
    void fill_Y(int spin, std::vector<cd>& out) const {
        out.assign(Ysize(ell_max), cd(0.0, 0.0));
        // Build zg^|spin|
        cd zg_abs_s(1.0, 0.0);
        for (int k = 0; k < std::abs(spin); ++k) zg_abs_s *= _last_zg;
        _fill_spin(spin, zg_abs_s, out);
    }

    //  spin defaults to mp_max (the value originally passed to constructor).
    void compute(const std::array<double,4>& R, std::vector<cd>& out) {
        compute_H(R);
        fill_Y(_spin_default, out);
    }
    std::vector<cd> compute(const std::array<double,4>& R) {
        std::vector<cd> out;
        compute(R, out);
        return out;
    }

private:
    mutable WignerH       _H;
    mutable std::vector<cd> _za_pow;
    mutable cd            _last_zg{1.0, 0.0};
    const int             _spin_default;

    void _fill_spin(int spin, cd zg_abs_s, std::vector<cd>& Y) const {
        const auto& Hwedge = _H.Hwedge;
        cd c1;
        if (spin >= 0) c1 = std::conj(zg_abs_s);
        else { double sgn=((-spin)&1)?-1.0:1.0; c1=sgn*zg_abs_s; }

        const int ell0 = std::abs(spin);
        for (int ell = ell0; ell <= ell_max; ++ell) {
            cd c2 = c1 * std::sqrt((2.0*ell+1.0)*INV_4PI);
            int iY = sylm::Yindex(ell, -ell);
            for (int m = -ell; m < 0; ++m) {
                int iH = _H.hindex(ell, m, -spin);
                Y[iY++] = c2 * Hwedge[iH] * std::conj(_za_pow[-m]);
            }
            for (int m = 0; m <= ell; ++m) {
                int iH = _H.hindex(ell, m, -spin);
                Y[iY++] = c2 * epsilon(m) * Hwedge[iH] * _za_pow[m];
            }
        }
    }
};

inline std::vector<cd> Ylm(int ell_max, int s, double theta, double phi) {
    double ct=std::cos(theta/2), st=std::sin(theta/2);
    double cp=std::cos(phi/2),   sp=std::sin(phi/2);
    std::array<double,4> R={ct*cp, +st*sp, st*cp, ct*sp};
    SphericalHarmonics sh(ell_max, std::abs(s));
    return sh.compute(R);
}

} // namespace sylm