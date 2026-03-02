#include <Eigen/Dense>
#include <vector>
#include <iostream>

#include "spin_weighted_sh.hpp"

using namespace std;
using cd = complex<double>;
const double PI = 3.14159265358979323846;

struct vect {
    double Yp2_re;
    double Yp2_im;
    double Ym2_re;
    double Ym2_im;
};

extern "C" {
    vect swsh_test(double theta, double phi, int ell, int m) {

        std::vector<double> theta_nodes;
        std::vector<double> phi_nodes;
        theta_nodes.push_back(theta);
        phi_nodes.push_back(phi);

        std::vector<cd> arr_p2 = sylm::Ylm(ell, +2, theta, phi);
        std::vector<cd> arr_m2 = sylm::Ylm(ell, -2, theta, phi);

        int mindex = ell*(ell+1)+m;

        vect result;

        result.Yp2_re = arr_p2[mindex].real();
        result.Yp2_im = arr_p2[mindex].imag();
        result.Ym2_re = arr_m2[mindex].real();
        result.Ym2_im = arr_m2[mindex].imag();

        return result;

    }

}

int main() {
    return 1;
}