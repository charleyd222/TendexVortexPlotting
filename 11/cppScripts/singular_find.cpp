vect singular_find(
    double r_val,
    int icity,
    double ending_tolerance,
    double delta_0,
    double safety,
    double h0,
    vis_params model_param,
    double dPhi,
    double dTheta,
    int section_it,
    double phi_min,
    double phi_max,
    double theta_min,
    double theta_max
) {
    int count = 0;
    vect r;

    for (double t = theta_min; t <= theta_max; t += dTheta) {
        for (double p = phi_min; p <= phi_max; p += dPhi) {

            double rad[4] = {-0.5 * M_PI, 0.0, 0.5 * M_PI, M_PI};
            double pointsPhi[4] = {
                p + 0.5 * dPhi * std::cos(rad[0]),
                p + 0.5 * dPhi * std::cos(rad[1]),
                p + 0.5 * dPhi * std::cos(rad[2]),
                p + 0.5 * dPhi * std::cos(rad[3])
            };
            double pointsTheta[4] = {
                t + 0.5 * dTheta * std::sin(rad[0]),
                t + 0.5 * dTheta * std::sin(rad[1]),
                t + 0.5 * dTheta * std::sin(rad[2]),
                t + 0.5 * dTheta * std::sin(rad[3])
            };

            double dotsX[4];
            double dotsY[4];
            bool found = false;

            for (int i = 0; i < 4; ++i) {
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
                    dotsX[i] = x0;
                    dotsY[i] = y0;
                    found = true;
                }
            }

            if (found) {
                double v0x = dotsX[0];
                double v0y = dotsY[0];
                double v1x = dotsX[1];
                double v1y = dotsY[1];
                double v2x = dotsX[2];
                double v2y = dotsY[2];
                double v3x = dotsX[3];
                double v3y = dotsY[3];

                double dot1 =
                    (v0x*v2x + v0y*v2y) /
                    (std::hypot(v0x, v0y) * std::hypot(v2x, v2y));
                double dot2 =
                    (v1x*v3x + v1y*v3y) /
                    (std::hypot(v1x, v1y) * std::hypot(v3x, v3y));

                if (dot1 > 1.0) dot1 = 1.0;
                if (dot1 < -1.0) dot1 = -1.0;
                if (dot2 > 1.0) dot2 = 1.0;
                if (dot2 < -1.0) dot2 = -1.0;

                double angle1 = std::acos(dot1);
                double angle2 = std::acos(dot2);

                if (std::abs(angle2) == 0.0) {
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