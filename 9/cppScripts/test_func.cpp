#include <iostream>
#include <Eigen>
#include <cfloat>
#include <cmath>
 
using namespace std;
using namespace Eigen;

Matrix3f f2D(const Vector3f& r_V, double sepX, double sepY, double sinT, double cosT) {
    Eigen::Matrix3f e1;
    Eigen::Matrix3f e2;
    Eigen::Matrix3f e3;
    Eigen::Matrix3f e4; 
    Eigen::Matrix3f E;
    Eigen::Matrix3f eTrace;

    double x0 = r_V(0) - sepX;
    double x1 = r_V(1) - sepY;
    double x2 = r_V(2);
    double r2 = (x0 * x0 + x1 * x1);
    double r5 = std::pow(x0 * x0 + x1 * x1, 2.5);

    double I_diag = (cosT * cosT) - (sinT * sinT);
    double I_off = 2 * cosT * sinT;
    double IpqXpXq = (I_diag * x0 * x0) + 2 * (I_off * x0 * x1) + (-I_diag * x1 * x1);
           
    e1 << I_diag, I_off, 0,
        I_off, -I_diag, 0,
        0, 0, 0;

    e2 << (x0 * x0 * I_diag) + (x0 * x1 * I_off), .5 * x0 * (x0 * I_off  + x1 * I_diag) + .5 * x1 * (x0 * -I_diag  + x1 * I_off), 0,
            .5 * x0 * (x0 * I_off  + x1 * I_diag) + .5 * x1 * (x0 * -I_diag  + x1 * I_off), (x0 * x1 * I_off) + (x1 * x1 * -I_diag), 0,
            0, 0, 0;
   
    // Const
    e3 << IpqXpXq, 0.0, 0.0,
          0.0, IpqXpXq, 0.0,
          0.0, 0.0, IpqXpXq;

    e4 << x0 * x0 * IpqXpXq, x0 * x1 * IpqXpXq, 0.0,
          x0 * x1 * IpqXpXq, x1 * x1 * IpqXpXq, 0.0,
          0.0, 0.0, 0.0;

    E = ((6 * e1) / (r5) - (60 * e2 + 15 * e3) / (r5 * r2) + (105 * e4) / (r5 * r2 * r2));

    double trace = E(0,0) + E(1,1) + E(2,2);
    eTrace << (1/3)*trace, 0.0, 0.0,
              0.0, (1/3)*trace, 0.0,
              0.0, 0.0, (1/3)*trace;
    //double norm = std::pow(r_V.norm(), 5);
    //eNorm << norm, norm, norm, 
    //         norm, norm, norm,
    //         norm, norm, norm;
    E = (E - eTrace);
    //E = E.cwiseProduct(eNorm);
    
    return E;
}

Matrix3f f3D(const Vector3f& r_V, double sepX, double sepY, double sinT, double cosT) {
    Eigen::Matrix3f e1;
    Eigen::Matrix3f e2;
    Eigen::Matrix3f e3;
    Eigen::Matrix3f e4;
    Eigen::Matrix3f e5;
    Eigen::Matrix3f E;
    Eigen::Matrix3f eTrace;

    double x0 = r_V(0) + sepX;
    double x1 = r_V(1) + sepY;
    double x2 = r_V(2);
    double r2 = (x0 * x0 + x1 * x1 + x2 * x2);
    double r5 = std::pow(x0 * x0 + x1 * x1 + x2 * x2, 2.5);
    
    double I_diag = (cosT * cosT) - (sinT * sinT);
    double I_off = 2 * cosT * sinT;
    double IpqXpXq = (I_diag * x0 * x0) + 2 * (I_off * x0 * x1) + (-I_diag * x1 * x1);
           
    e1 << I_diag, I_off, 0,
          I_off, -I_diag, 0,
          0, 0, 0;

    e2 << (x0 * x0 * I_diag) + (x0 * x1 * I_off), .5 * x0 * (x0 * I_off  + x1 * I_diag) + .5 * x1 * (x0 * -I_diag  + x1 * I_off), 0,
          .5 * x0 * (x0 * I_off  + x1 * I_diag) + .5 * x1 * (x0 * -I_diag  + x1 * I_off), (x0 * x1 * I_off) + (x1 * x1 * -I_diag), 0,
          0, 0, 0;
   
    // Const
    e3 << IpqXpXq, 0.0, 0.0,
          0.0, IpqXpXq, 0.0,
          0.0, 0.0, IpqXpXq;

    e4 << x0 * x0 * IpqXpXq, x1 * x0 * IpqXpXq, x2 * x0 * IpqXpXq,
          x0 * x1 * IpqXpXq, x1 * x1 * IpqXpXq, x2 * x1 * IpqXpXq,
          x0 * x2 * IpqXpXq, x1 * x2 * IpqXpXq, x2 * x2 * IpqXpXq;

    e5 << 0.0, 0.0, x2 * (I_diag * x0 + I_off * x1),
          0.0, 0.0, x2 * (I_off * x0 - I_diag * x1),
          x2 * (I_diag * x0 + I_off * x1), x2 * (I_off * x0 - I_diag * x1), 0.0;

    E = ((6 * e1) / (r5) - (60 * e2 + 30 * e5 + 15 * e3) / (r5 * r2) + (105 * e4) / (r5 * r2 * r2));


    double trace = E(0,0) + E(1,1) + E(2,2);
    eTrace << (1/3)*trace, 0.0, 0.0,
              0.0, (1/3)*trace, 0.0,
              0.0, 0.0, (1/3)*trace;

    E = (E - eTrace);
    
    return E;
}

extern "C" {

struct mat {
    double val[3][3];
};

mat test_func(double x, double y, double z, double sepX, double sepY, double theta) {
    mat m;
    Matrix3f result;
    Vector3f r_V;

    r_V << x, y, z;

    double sinT = sin(theta);
    double cosT = cos(theta);

    result = f3D(r_V, sepX, sepY, sinT, cosT);

    m.val[0][0] = result(0,0);
    m.val[0][1] = result(0,1);
    m.val[0][2] = result(0,2);
    m.val[1][0] = result(1,0);
    m.val[1][1] = result(1,1);
    m.val[1][2] = result(1,2);
    m.val[2][0] = result(2,0);
    m.val[2][1] = result(2,1);
    m.val[2][2] = result(2,2);

    return m;
}

}

int main() {
    return 0;
}