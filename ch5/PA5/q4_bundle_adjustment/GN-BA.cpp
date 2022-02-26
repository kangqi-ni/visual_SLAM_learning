#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "sophus/se3.hpp"

using namespace std;

typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>> VecVector3d;
typedef vector<Vector2d, Eigen::aligned_allocator<Vector3d>> VecVector2d;
typedef Matrix<double, 6, 1> Vector6d;

string p3d_file = "./p3d.txt";
string p2d_file = "./p2d.txt";

int main(int argc, char **argv) {
    // 2d and 3d points
    VecVector2d p2d;
    VecVector3d p3d;
    
    // Intrinsic
    Matrix3d K;
    double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    // Load points in to p2d
    ifstream fin(p2d_file);
    if (!fin) {
        cerr << "Couldn't read file: " << p2d_file << endl;
        return 1;
    }
    while (!fin.eof()) {
        double x,y;
        fin >> x >> y;
        Vector2d p(x,y);
        p2d.push_back(p);
    }
    fin.close();   

    // Load points into p3d
    fin.open(p3d_file);
    if (!fin) {
        cerr << "Couldn't read file: " << p3d_file << endl;
        return 1;
    }
    while (!fin.eof()) {
        double x,y,z;
        fin >> x >> y >> z;
        Vector3d p(x,y,z);
        p3d.push_back(p);
    }
    fin.close();

    assert(p3d.size() == p2d.size());

    int iterations = 100;
    double cost = 0, lastCost = 0;
    int nPoints = p3d.size();
    cout << "points: " << nPoints << endl;

    Sophus::SE3d T_esti; // estimated pose

    for (int iter = 0; iter < iterations; iter++) {
        // Hessian matrix
        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // Compute cost
        for (int i = 0; i < nPoints; i++) {
            // Compute cost for p3d[i] and p2d[i]
            Vector3d p_cam = T_esti * p3d[i];
            Vector3d p_reprojected = K * p_cam/p_cam[2];
            Vector2d e = p2d[i] - Vector2d(p_reprojected[0], p_reprojected[1]);
            cost += e.squaredNorm();

	        // Compute jacobian
            double X = p_cam[0], Y = p_cam[1], Z = p_cam[2];
            Matrix<double, 2, 6> J;
            J(0,0) = fx/Z;
            J(0,1) = 0;
            J(0,2) = -fx*X/(Z*Z);
            J(0,3) = -fx*X*Y/(Z*Z);
            J(0,4) = fx + fx*X*X/(Z*Z);
            J(0,5) = -fx*Y/Z;

            J(1,0) = 0;
            J(1,1) = fy*Z;
            J(1,2) = -fy*Y/(Z*Z);
            J(1,3) = -fy - fy*Y*Y/(Z*Z);
            J(1,4) = fy*X*Y/(Z*Z);
            J(1,5) = fy*X/Z;
            J = -J;

            // Compute H and b
            H += J.transpose() * J;
            b += -J.transpose() * e;
        }

	    // Solve dx 
        Vector6d dx = H.ldlt().solve(b);

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // Cost increases
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // Update estimation
        T_esti = Sophus::SE3d::exp(dx) * T_esti;
        
        lastCost = cost;

        cout << "iteration " << iter << " cost=" << cout.precision(12) << cost << endl;
    }

    cout << "estimated pose: \n" << T_esti.matrix() << endl;
    return 0;
}
