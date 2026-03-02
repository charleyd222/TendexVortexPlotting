#include <Eigen/Dense>


Eigen::Vector3d superPoynting(
    const Eigen::Matrix3d& E,
    const Eigen::Matrix3d& B
) {
    Eigen::Vector3d S;
    S.setZero();

    // Levi-Civita symbol
    auto eps = [](int i, int j, int k) {
        if ((i==0 && j==1 && k==2) ||
            (i==1 && j==2 && k==0) ||
            (i==2 && j==0 && k==1)) return  1.0;
        if ((i==0 && j==2 && k==1) ||
            (i==2 && j==1 && k==0) ||
            (i==1 && j==0 && k==2)) return -1.0;
        return 0.0;
    };

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                    S(i) += eps(i,j,k) * E(j,l) * B(k,l);

    return S;

}