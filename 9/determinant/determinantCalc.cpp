#include <iostream>
#include <Eigen>
#include <cfloat>
#include <cmath>
 
using namespace std;
using namespace Eigen;

extern "C" {
    
double f1Quadropole(double x0, double x1) {
    Eigen::Matrix3f e1;
    Eigen::Matrix3f e2;
    Eigen::Matrix3f e3;
    Eigen::Matrix3f e4;
    Eigen::Matrix3f E;
    double x2 = 0;
    double r2 = (x0 * x0 + x1 * x1 + x2 * x2);
    
    double aI = 1.0;
    double IpqXpXq = (aI * (x0 * x0 - x1 * x1));

    e1 << aI, 0.0, 0.0,
        0.0, -1 * aI, 0.0,
        0.0, 0.0, 0.0;

    e2 << 2 * aI * x0 * x0, 0.0, 0.0,
          0.0, -2 * aI * x1 * x1, 0.0,
          0.0, 0.0, 0.0;

    e3 << IpqXpXq, 0.0, 0.0,
          0.0, IpqXpXq, 0.0,
          0.0, 0.0, IpqXpXq;

    e4 << x0 * x0 * IpqXpXq, x0 * x1 * IpqXpXq, 0.0,
          x0 * x1 * IpqXpXq, x1 * x1 * IpqXpXq, 0.0,
          0.0, 0.0, 0.0;
          
    E = (-6 * e1) + (30 * e2 + 15 * e3) / (r2) + (-105 * e4 / (r2*r2));

    return E.determinant();

    
}

double fCustomT2Quadropole(double x0, double x1, double sepX, double sepY, double theta) {
    double sinT = sin(theta);
    double cosT = cos(theta);
    Eigen::Matrix3f e01;
    Eigen::Matrix3f e02;
    Eigen::Matrix3f e03;
    Eigen::Matrix3f e04;
    Eigen::Matrix3f e11;
    Eigen::Matrix3f e12;
    Eigen::Matrix3f e13;
    Eigen::Matrix3f e14;    
    Eigen::Matrix3f E;

    double x00 = x0 + sepX;
    double x01 = x1;// + sepY;
    double x02 = 0;
    
    double x10 = x0 - sepX;
    double x11 = x1;// - sepY;
    double x12 = 0;
    double r02 = (x00 * x00 + x01 * x01);
    double r05 = std::pow(x00 * x00 + x01 * x01, 2.5);
    double r12 = (x10 * x10 + x11 * x11);
    double r15 = std::pow(x10 * x10 + x11 * x11, 2.5);
    
    double aI = 1.0;
    double IpqXpXq0 = ((x00 * x00 - x01 * x01));


    e01 << 1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
        0.0, 0.0, 0.0;
    
    e02 << (x00 * x00), 0.0, 0.0,
        0.0, -1.0 * (x01 * x01), 0.0,
        0.0, 0.0, 0.0;

    e03 << IpqXpXq0, 0.0, 0.0,
        0.0, IpqXpXq0, 0.0,
        0.0, 0.0, IpqXpXq0;

    e04 << x00 * x00 * IpqXpXq0, x00 * x01 * IpqXpXq0, 0.0,
        x00 * x01 * IpqXpXq0, x01 * x01 * IpqXpXq0, 0.0,
        0.0, 0.0, 0.0;
    
    double I11 = (cosT * cosT) - (sinT * sinT);
    double I12 = 2 * cosT * sinT;
    double I22 = (sinT * sinT) - (cosT * cosT);
    double IpqXpXq1 = (I11 * x10 * x10) + 2 * (I12 * x10 * x11) + (I22 * x11 * x11);
           
    e11 << I11, I12, 0,
        I12, I22, 0,
        0, 0, 0;

    e12 << (x10 * x10 * I11) + (x10 * x11 * I12), .5 * x10 * (x10 * I12  + x11 * I11) + .5 * x11 * (x10 * I22  + x11 * I12), 0,
            .5 * x10 * (x10 * I12  + x11 * I11) + .5 * x11 * (x10 * I22  + x11 * I12), (x10 * x11 * I12) + (x11 * x11 * I22), 0,
            0, 0, 0;
   
    // Const
    e13 << IpqXpXq1, 0.0, 0.0,
            0.0, IpqXpXq1, 0.0,
            0.0, 0.0, IpqXpXq1;

    e14 << x10 * x10 * IpqXpXq1, x10 * x11 * IpqXpXq1, 0.0,
            x10 * x11 * IpqXpXq1, x11 * x11 * IpqXpXq1, 0.0,
            0.0, 0.0, 0.0;
    
    E = ((-6 * e01) / (r05) + (60 * e02 + 15 * e03) / (r05 * r02) + (-105 * e04) / (r05 * r02 * r02)) + 
        ((-6 * e11) / (r15) + (60 * e12 + 15 * e13) / (r15 * r12) + (-105 * e14) / (r15 * r12 * r12));
    
    double trace = E(0,0) + E(1,1) + E(2,2);
    double symmetric = (E(0,1) - E(1,0)) + (E(0,2) - E(2,0)) + (E(1,2) - E(2,1));
    //std::cout << sinT << " " << cosT << '\n';
    //std::cout << trace << " " << symmetric << '\n';

    return (E).determinant();

    
}

};
int main() {
    //rka_iter(1.0, 1.0, 1.0, 2000, 1, .5);

    return 0;
}
