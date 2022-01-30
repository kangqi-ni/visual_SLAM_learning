#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

int main(int argc, char** argv) {
    Eigen::Quaterniond q_wr (0.55, 0.3, 0.2, 0.2); 
    Eigen::Vector3d t_wr (0.1, 0.2, 0.3);

    Eigen::Quaterniond q_rb (0.99, 0, 0, 0.01);
    Eigen::Vector3d t_rb (0.05, 0, 0.5);

    Eigen::Quaterniond q_bl (0.3, 0.5, 0, 20.1);
    Eigen::Vector3d t_bl (0.4, 0, 0.5);

    Eigen::Quaterniond q_bc (0.8, 0.2, 0.1, 0.1);
    Eigen::Vector3d t_bc (0.5, 0.1, 0.5);

    // Normalize quaternions 
    q_wr.normalize();
    q_rb.normalize();
    q_bl.normalize();
    q_bc.normalize();

    // point in camera frame
    Eigen::Vector3d p_c (0.3, 0.2, 1.2);
    std::cout << "point in cameram frame: " << p_c.transpose() << '\n';

    // Transform point from camera frame to lidar frame
    // T_lb * T_bc * p_c
    Eigen::Isometry3d T_bc (q_bc);
    T_bc.pretranslate(t_bc);
    Eigen::Isometry3d T_bl (q_bl);
    T_bl.pretranslate(t_bl);
    Eigen::Vector3d p_l = T_bl.inverse() * T_bc * p_c;
    std::cout << "point in lidar frame: " << p_l.transpose() << '\n';

    // Transform point from camera frame to world frame
    // T_wr * T_rb * T_bc * p_c
    Eigen::Isometry3d T_rb (q_rb);
    T_rb.pretranslate(t_rb);
    Eigen::Isometry3d T_wr (q_wr);
    T_wr.pretranslate(t_wr);
    Eigen::Vector3d p_w = T_wr * T_rb * T_bc * p_c;
    std::cout << "point in world frame: " << p_w.transpose() << '\n';

    return 0;
}