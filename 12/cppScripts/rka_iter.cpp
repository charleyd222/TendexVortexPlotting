#include <iostream>
#include <Eigen>
#include <cfloat>
#include <complex>
#include <cmath>
#include <limits>
#include <boost/math/special_functions/hypergeometric_pFq.hpp>

using namespace std;
using namespace Eigen;


struct vis_params {
    double M;
    double omega1;
    double omega2;
    double t;
    double C2;
    double w2;
    int ell;
};

struct vect {
    double x[10000];
    double y[10000];
    double z[10000];
    double m[10000];
    int its;
};

struct mat {
    double m11;
    double m12;
    double m21;
    double m22;
};

inline double factorial(int n)
{
    return std::tgamma(n + 1.0);
}

inline double binom(int n, int k)
{
    if (k < 0 || k > n) return 0.0;
    return factorial(n) / (factorial(k) * factorial(n - k));
}


std::complex<double> sYlm(int s, int l, int m, double theta, double phi) {
    
    std::complex<double> summation = 0.0;
    std::complex<double> phase =
            std::exp(1i * static_cast<double>(m) * phi);

    for (int r = 0; r <= l - s; ++r)
    {
        double coeff =
            binom(l - s, r) *
            binom(l + s, r + s - m) *
            std::pow(-1.0, l - r - s);        

        int p = 2*r + s - m;

        double term =
            pow(1/tan(theta/2), p);

        summation += coeff * phase * term;

    }

    double prefactor =
        std::pow(-1.0, m) *
        std::sqrt(
            factorial(l + m) * factorial(l - m) * (2.0 * l + 1.0) /
            (factorial(l + s) * factorial(l - s) * 4.0 * M_PI)
        ) * 
        pow(sin(theta / 2), 2 * l);

    std::complex<double> Y =
        prefactor *
        summation;

    // nan_to_num equivalent
    if (!std::isfinite(Y.real()) || !std::isfinite(Y.imag()))
        Y = 0.0;

    return Y;
}

Matrix2cd TE2lm_calc(const std::complex<double>& Y2, const std::complex<double>& Yn2) {
    std::complex<double> a = Yn2 + Y2;
    std::complex<double> b = 1i * (Yn2 - Y2);

    Matrix2cd E;
    E << a,  b,
         b, -a;

    E *= std::pow(2.0, -1.5);

    return E;
}

Matrix2cd TB2lm_calc(const std::complex<double>& Y2, const std::complex<double>& Yn2) {
    std::complex<double> a = Yn2 - Y2;
    std::complex<double> b = 1i * (Yn2 + Y2);

    Matrix2cd E;
    E << a,  b,
         b, -a;

    E *= (-1i) / std::pow(2.0, 1.5);

    return E;
}

Matrix3cd EleM(const Vector3d& r_V, double M, double t, int l, int m, double Omega, double omega) {
    double r = r_V.norm();
    complex I = pow(Omega, l+2) * M * std::exp(-1i * (Omega * (t-r_V(0)) + omega));
    Matrix3cd E;
    Matrix2cd TE2;

    complex Y2 = sYlm(2, l, m, r_V(1), r_V(2));
    complex Yn2 = sYlm(-2, l, m, r_V(1), r_V(2));


    TE2 = TE2lm_calc(Y2, Yn2);
    //TE2 = I * TE2;
    E(0,0) = 0.0;
    E(0,1) = 0.0;
    E(0,2) = 0.0;
    E(1,0) = 0.0;
    E(2,0) = 0.0;
    E(1,1) = TE2(0,0);
    E(1,2) = TE2(0,1);
    E(2,1) = TE2(1,0);
    E(2,2) = TE2(1,1);

    return E;

}

Matrix3cd EleC(const Vector3d& r_V, double M, double t, int l, int m, double Omega, double omega) {
    double r = r_V.norm();
    complex S = pow(Omega, l+2) * M * std::exp(-1i * (Omega * (t-r_V(0)) + omega));
    Matrix3cd E;
    Matrix2cd TB2;

    complex Y2 = sYlm(2, l, m, r_V(1), r_V(2));
    complex Yn2 = sYlm(-2, l, m, r_V(1), r_V(2));

    TB2 = TB2lm_calc(Y2, Yn2);
    TB2 = S * TB2;
    E(0,0) = 0.0;
    E(0,1) = 0.0;
    E(0,2) = 0.0;
    E(1,0) = 0.0;
    E(2,0) = 0.0;
    E(1,1) = TB2(0,0);
    E(1,2) = TB2(0,1);
    E(2,1) = TB2(1,0);
    E(2,2) = TB2(1,1);

    return E;
}

Matrix3cd E0(const Vector3d& r_V, double lambda, double dTheta) {
    double phi = r_V(2);
    double theta = r_V(1);
    Matrix3cd E0;

    complex exponential = std::exp(-1 * pow(theta,2) / (2 * dTheta));

    complex a = cos(2 * phi) * pow(sin(theta), 2);
    complex b = cos(theta) * sin(2 * phi);

    E0 << 0,0,0,
          0,a,b,
          0,b,-a;

    E0 = E0 * exponential;

    return E0;
}

Matrix3d fN(const Vector3d& r_V, vis_params vis_params) {
    double M = vis_params.M;
    double C = vis_params.C2;
    double omega1 = vis_params.omega1;
    double omega2 = vis_params.omega2;
    double t = vis_params.t;
    //double t = r_V(0);
    double w = vis_params.w2;

    int l = vis_params.ell;
    
    return E0(r_V, M, C).real();
    //return (EleM(r_V, M, t, l, l, omega1, 0) + C * EleC(r_V, M, t, l, l, omega2, w)).real();
}

Matrix3d f(const Vector3d& r_V, vis_params vis_params) {
    double M = vis_params.M;
    double C = vis_params.C2;
    double omega1 = vis_params.omega1;
    double omega2 = vis_params.omega2;
    double t = vis_params.t;
    //double t = r_V(0);
    double w = vis_params.w2;
    double l_max = vis_params.ell;

    vector<int> l = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100};
    vector<int> m = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2};
    vector<double> A = {0.290397,0.264786,0.203361,0.123797,0.043046,-0.0263431,-0.0780927,-0.111639,-0.130006,-0.137491,-0.13806,-0.134676,-0.129269,-0.122983,-0.116453,-0.110016,-0.103845,-0.0980209,-0.092573,-0.0875043,-0.0828024,-0.0784473,-0.0744154,-0.070682,-0.067223,-0.0640151,-0.0610369,-0.0582684,-0.0556915,-0.0532895,-0.0510474,-0.0489516,-0.04699,-0.0451512,-0.0434254,-0.0418035,-0.0402771,-0.038839,-0.0374823,-0.036201,-0.0349894,-0.0338425,-0.0327557,-0.0317248,-0.0307458,-0.0298154,-0.0289302,-0.0280873,-0.0272839,-0.0265176,-0.025786,-0.025087,-0.0244187,-0.0237791,-0.0231667,-0.0225799,-0.0220171,-0.0214772,-0.0209587,-0.0204605,-0.0199816,-0.019521,-0.0190776,-0.0186506,-0.0182392,-0.0178426,-0.01746,-0.0170909,-0.0167344,-0.0163901,-0.0160574,-0.0157357,-0.0154246,-0.0151235,-0.014832,-0.0145497,-0.0142762,-0.0140111,-0.0137541,-0.0135048,-0.0132629,-0.0130281,-0.0128001,-0.0125787,-0.0123635,-0.0121544,-0.0119511,-0.0117534,-0.011561,-0.0113738,-0.0111916,-0.0110142,-0.0108414,-0.0106731,-0.010509,-0.0103491,-0.0101932,-0.0100412,-0.0098929,0.290397,0.264786,0.203361,0.123797,0.043046,-0.0263431,-0.0780927,-0.111639,-0.130006,-0.137491,-0.13806,-0.134676,-0.129269,-0.122983,-0.116453,-0.110016,-0.103845,-0.0980209,-0.092573,-0.0875043,-0.0828024,-0.0784473,-0.0744154,-0.070682,-0.067223,-0.0640151,-0.0610369,-0.0582684,-0.0556915,-0.0532895,-0.0510474,-0.0489516,-0.04699,-0.0451512,-0.0434254,-0.0418035,-0.0402771,-0.038839,-0.0374823,-0.036201,-0.0349894,-0.0338425,-0.0327557,-0.0317248,-0.0307458,-0.0298154,-0.0289302,-0.0280873,-0.0272839,-0.0265176,-0.025786,-0.025087,-0.0244187,-0.0237791,-0.0231667,-0.0225799,-0.0220171,-0.0214772,-0.0209587,-0.0204605,-0.0199816,-0.019521,-0.0190776,-0.0186506,-0.0182392,-0.0178426,-0.01746,-0.0170909,-0.0167344,-0.0163901,-0.0160574,-0.0157357,-0.0154246,-0.0151235,-0.014832,-0.0145497,-0.0142762,-0.0140111,-0.0137541,-0.0135048,-0.0132629,-0.0130281,-0.0128001,-0.0125787,-0.0123635,-0.0121544,-0.0119511,-0.0117534,-0.011561,-0.0113738,-0.0111916,-0.0110142,-0.0108414,-0.0106731,-0.010509,-0.0103491,-0.0101932,-0.0100412,-0.0098929};
    vector<double> B = {0.293507,0.280348,0.245926,0.208078,0.178072,0.159602,0.150825,0.147607,0.146125,0.144037,0.140469,0.135479,0.129512,0.123049,0.116469,0.11002,0.103846,0.098021,0.092573,0.0875043,0.0828024,0.0784473,0.0744154,0.070682,0.067223,0.0640151,0.0610369,0.0582684,0.0556915,0.0532895,0.0510474,0.0489516,0.04699,0.0451512,0.0434254,0.0418035,0.0402771,0.038839,0.0374823,0.036201,0.0349894,0.0338425,0.0327557,0.0317248,0.0307458,0.0298154,0.0289302,0.0280873,0.0272839,0.0265176,0.025786,0.025087,0.0244187,0.0237791,0.0231667,0.0225799,0.0220171,0.0214772,0.0209587,0.0204605,0.0199816,0.019521,0.0190776,0.0186506,0.0182392,0.0178426,0.01746,0.0170909,0.0167344,0.0163901,0.0160574,0.0157357,0.0154246,0.0151235,0.014832,0.0145497,0.0142762,0.0140111,0.0137541,0.0135048,0.0132629,0.0130281,0.0128001,0.0125787,0.0123635,0.0121544,0.0119511,0.0117534,0.011561,0.0113738,0.0111916,0.0110142,0.0108414,0.0106731,0.010509,0.0103491,0.0101932,0.0100412,0.0098929,-0.293507,-0.280348,-0.245926,-0.208078,-0.178072,-0.159602,-0.150825,-0.147607,-0.146125,-0.144037,-0.140469,-0.135479,-0.129512,-0.123049,-0.116469,-0.11002,-0.103846,-0.098021,-0.092573,-0.0875043,-0.0828024,-0.0784473,-0.0744154,-0.070682,-0.067223,-0.0640151,-0.0610369,-0.0582684,-0.0556915,-0.0532895,-0.0510474,-0.0489516,-0.04699,-0.0451512,-0.0434254,-0.0418035,-0.0402771,-0.038839,-0.0374823,-0.036201,-0.0349894,-0.0338425,-0.0327557,-0.0317248,-0.0307458,-0.0298154,-0.0289302,-0.0280873,-0.0272839,-0.0265176,-0.025786,-0.025087,-0.0244187,-0.0237791,-0.0231667,-0.0225799,-0.0220171,-0.0214772,-0.0209587,-0.0204605,-0.0199816,-0.019521,-0.0190776,-0.0186506,-0.0182392,-0.0178426,-0.01746,-0.0170909,-0.0167344,-0.0163901,-0.0160574,-0.0157357,-0.0154246,-0.0151235,-0.014832,-0.0145497,-0.0142762,-0.0140111,-0.0137541,-0.0135048,-0.0132629,-0.0130281,-0.0128001,-0.0125787,-0.0123635,-0.0121544,-0.0119511,-0.0117534,-0.011561,-0.0113738,-0.0111916,-0.0110142,-0.0108414,-0.0106731,-0.010509,-0.0103491,-0.0101932,-0.0100412,-0.0098929};
    Matrix3cd T;
    Matrix3cd T_Temp;
    T << 0,0,0,
         0,0,0,
         0,0,0;

    for (int i = 0; i < l_max; i++) {
        T_Temp = A[i] * EleM(r_V, M, t, l[i], m[i], omega1, 0) + (1i * B[i]) * EleC(r_V, M, t, l[i], m[i], omega2, w);

        T = T + T_Temp;
    }

    return T.real();
}

double eigen_solve_val(Matrix3d E_temp, int icity) {
    // Compute eigenvalues and eigenvectors
    EigenSolver<Matrix3d> solver(E_temp);
    Vector3cd eigenvalues = solver.eigenvalues();   // Complex eigenvalues

    double result = eigenvalues.real().maxCoeff();
    return result;
    
}

// Function to get the eigenvector corresponding to the largest eigenvalue with the given sign
Vector3d eigen_solve(Matrix3d E_temp, int icity) {
    // Compute eigenvalues and eigenvectors
    EigenSolver<Matrix3d> solver(E_temp);
    Vector3cd eigenvalues = solver.eigenvalues();   // Complex eigenvalues
    Matrix3cd eigenvectors = solver.eigenvectors(); // Corresponding eigenvectors
    Vector3d result;

    double maxEigenvalue = -1e6; // Large negative value for initialization
    int maxIndex = -1;
    // Loop through eigenvalues to find the largest with the same sign as `icity`
    for (int i = 0; i < 3; ++i) {
        double realVal = eigenvalues[i].real(); // Extract real part of eigenvalue
        
        if (icity * realVal > 0 && realVal * icity > maxEigenvalue) {
            maxEigenvalue = realVal * icity;
            maxIndex = i;
        }
    }
    // Convert complex eigenvector to real vector (assuming it's real-valued)
    if (maxIndex < 0) {
        result << 0.0,0.0,0.0;
        return result;
    } 
    
    result = eigenvectors.col(maxIndex).real();
    
    return result / result.norm();
}

extern "C" {

double val_return(double x, double y, double z, int icity, vis_params vals, int spherical= 0) {
    Vector3d r;
    if (spherical == 1) {
        r = {x, y, z};
    } else {
        double r_val = sqrt(x*x + y*y + z*z);
        if (abs(r_val) < 1e-6) {
            return 0.0;
        }
        r = {r_val, acos(z / r_val), atan2(y,x)};
    }

    return eigen_solve_val(f(r, vals), icity);
}

vect rka_iter(double seed_r, double seed_theta, double seed_phi, int num_its, int icity, double ending_tolerance, double delta_0, double safety, double h0, vis_params vals) {    
    static const double a2 = 1.0 / 5.0, a3 = 3.0 / 10.0, a4 = 3.0 / 5.0, a5 = 1.0, a6 = 7.0 / 8.0;
    static const double b21 = 1.0 / 5.0;
    static const double b31 = 1.0 / 40.0, b32 = 9.0 / 40.0;
    static const double b41 = 3.0 / 10.0, b42 = -9.0 / 10.0, b43 = 6.0 / 5.0;
    static const double b51 = -11.0 / 54.0, b52 = 5.0/2.0, b53 = -70.0 / 27.0, b54 = 35.0 / 27.0;
    static const double b61 = 1631.0 / 55296.0, b62 = 175.0 / 512.0, b63 = 575.0 / 13824.0, b64 = 44275.0 / 110592.0, b65 = 253.0 / 4096.0;
    static const double c1 = 37.0 / 378.0, c3 = 250.0 / 621.0, c4 = 125.0 / 594.0, c6 = 512.0 / 1771.0;
    static const double cd1 = 37.0 / 378.0 - 2825.0 / 27648.0, cd3 = 250.0 / 621.0 - 18575.0 / 48384.0, cd4 = 125.0 / 594.0 - 13525.0 / 55296.0, cd5 = -277.0 / 14336.0, cd6 = 512.0 / 1771.0 - 1.0 / 4.0;
    bool sizing = true;
    vect r_change_vect;

    Vector3d r = {seed_r, seed_theta, seed_phi};
    Vector3d r_past = {0.0, 0.0, 0.0};
    Vector3d r_change;
    int i;
    double h;

    for (i = 0; i < num_its; i++) {
        h = h0;
        sizing = true;

        while (sizing) {
            // Compute Runge-Kutta stages
            Matrix3d E1 = f(r, vals);
            Vector3d k1 = h * eigen_solve(E1, icity);

            Matrix3d E2 = f(r + b21 * k1, vals);
            Vector3d k2 = h * eigen_solve(E2, icity);

            Matrix3d E3 = f(r + b31 * k1 + b32 * k2, vals);
            Vector3d k3 = h * eigen_solve(E3, icity);

            Matrix3d E4 = f(r + b41 * k1 + b42 * k2 + b43 * k3, vals);
            Vector3d k4 = h * eigen_solve(E4, icity);

            Matrix3d E5 = f(r + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, vals);
            Vector3d k5 = h * eigen_solve(E5, icity);

            Matrix3d E6 = f(r + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, vals);
            Vector3d k6 = h * eigen_solve(E6, icity);

            Vector3d delta = cd1 * k1 + cd3 * k3 + cd4 * k4 + cd5 * k5 + cd6 * k6;
            
            if ((delta.array().abs() > delta_0).any()) {
                h *= safety * std::pow(std::abs(delta_0 / delta.norm()), 0.25f);
            } else {
                h *= safety * std::pow(std::abs(delta_0 / delta.norm()), 0.2f);

                r_change = c1 * k1 + c3 * k3 + c4 * k4 + c6 * k6;
                
                // Set step_sizing to false
                sizing = false;
            }
        }
        
        // Account for sign missmatch
        double r_mag = r_change.norm();
        double dot = r_past.dot(r_change/r_mag);

        if (dot < -.95) {
            r_change *= -1;
        }

        r += r_change;
        r_past = r_change / r_mag;
        double val = eigen_solve_val(f(r, vals), icity);

        r_change_vect.x[i] = r(0);
        r_change_vect.y[i] = r(1);
        r_change_vect.z[i] = r(2);
        r_change_vect.m[i] = val;

        if (abs(val) < ending_tolerance) {
            break;
        }
    
    }

    r_change_vect.its = i;


    return r_change_vect;
}

mat test_val(double x, double y, double z, vis_params vals, int spherical = 0) {
    Vector3d r;
    mat m;
    m.m11 = 0.0;
    m.m12 = 0.0;
    m.m21 = 0.0;
    m.m22 = 0.0;

    if (spherical == 0) {
        double r_val = sqrt(x*x + y*y + z*z);
        if (abs(r_val) < 1e-6) {
            r = {0,0,0};
        }
        else {
            r = {r_val, acos(z / r_val), atan2(y,x)};
        }
        
    }
    else {
        r = {x, y, z};
    }

    
    Matrix3d m_temp = f(r, vals);

    m.m11 = m_temp(1,1);
    m.m12 = m_temp(1,2);
    m.m21 = m_temp(2,1);
    m.m22 = m_temp(2,2);

    return m;
}

vect singular_find(double r_val, int icity, double ending_tolerance, double delta_0, double safety,
    double h0, vis_params model_param, double dPhi, double dTheta, int section_it, double phi_min,
    double phi_max, double theta_min, double theta_max
) {
    int count = 0;
    vect r;

    for (double t = theta_min; t <= theta_max; t += dTheta) {
        for (double p = phi_min; p <= phi_max; p += dPhi) {

            double rad[2] = {0.0, M_PI};
            double pointsPhi[2] = {
                p + 0.5 * dPhi * std::cos(rad[0]),
                p + 0.5 * dPhi * std::cos(rad[1])
            };
            double pointsTheta[2] = {
                t + 0.5 * dTheta * std::sin(rad[0]),
                t + 0.5 * dTheta * std::sin(rad[1])
            };

            std::vector<std::pair<double,double>> dots;

            for (int i = 0; i < 2; ++i) {
                vect v = rka_iter(
                    r_val,
                    pointsTheta[i],
                    pointsPhi[i],
                    section_it,
                    icity,
                    ending_tolerance,
                    delta_0,
                    safety,
                    h0,
                    model_param
                );

                if (v.its > section_it - 1) {
                    double y0 = v.y[0];
                    double x0 = v.z[0];
                    dots.emplace_back(y0, x0);
                }
            }

            if (dots.size() > 0) {
                double v0x = dots.front().first;
                double v0y = dots.front().second;
                double v1x = dots.back().first;
                double v1y = dots.back().second;

                double dot =
                    (v0x*v1x + v0y*v1y) /
                    (std::hypot(v0x, v0y) * std::hypot(v1x, v1y));

                if (dot > 1.0) dot = 1.0;
                if (dot < -1.0) dot = -1.0;

                double angle = std::acos(dot);

                if (std::abs(angle) == 0.0) {
                    r.x[count] = p;
                    r.y[count] = t;
                    count++;
                }
            }
        }
    }

    
    r.its = count;
    return r;
}

};

int main() {
    //rka_iter(1.0, 1.0, 1.0, 2000, 1, .5);

    return 0;
}
