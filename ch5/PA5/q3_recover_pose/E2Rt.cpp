#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace Eigen;

#include <sophus/so3.hpp>

#include <iostream>

using namespace std;

int main(int argc, char **argv) {
    // Essential matrix
    Matrix3d E;
    E << -0.0203618550523477, -0.4007110038118445, -0.03324074249824097,
            0.3939270778216369, -0.03506401846698079, 0.5857110303721015,
            -0.006788487241438284, -0.5815434272915686, -0.01438258684486258;

    // R,t to be recovered
    Matrix3d R;
    Vector3d t;

    // SVD and fix sigular values
    JacobiSVD<Matrix3d> svd (E, ComputeFullV | ComputeFullU);
    Matrix3d U = svd.matrixU();
    Matrix3d V = svd.matrixV();
    Vector3d singular_values = svd.singularValues();
    double singular_value = (singular_values[0] + singular_values[1]) / 2;
    singular_values = Vector3d(singular_value, singular_value, 0);
    DiagonalMatrix<double,3> sigma (singular_values);

    // set t1, t2, R1, R2 
    Matrix3d Rz1 = AngleAxisd(M_PI/2, Vector3d(0,0,1)).toRotationMatrix();
    Matrix3d Rz2 = AngleAxisd(-M_PI/2, Eigen::Vector3d(0,0,1)).toRotationMatrix();

    Matrix3d t_wedge1 = U * Rz1 * sigma * U.transpose(); 
    Matrix3d t_wedge2 = U * Rz2 * sigma * U.transpose();
    
    Matrix3d R1 = U * Rz1.transpose() * V.transpose();
    Matrix3d R2 = U * Rz2.transpose() * V.transpose();

    cout << "R1 = " << R1 << endl;
    cout << "R2 = " << R2 << endl;
    cout << "t1 = " << Sophus::SO3d::vee(t_wedge1) << endl;
    cout << "t2 = " << Sophus::SO3d::vee(t_wedge2) << endl;

    // check t^R=E up to scale
    Matrix3d tR = t_wedge1 * R1;
    cout << "t^R = " << tR << endl;

    return 0;
}