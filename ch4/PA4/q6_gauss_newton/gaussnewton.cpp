#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {
    double ar = 1.0, br = 2.0, cr = 1.0;         // real values
    double ae = 2.0, be = -1.0, ce = 5.0;        // estimations
    int N = 100;                                 // number of data points
    double w_sigma = 1.0;                        // sd of gaussian noise
    cv::RNG rng;                                 // OpenCV random generator

    // Generate data using y = exp(a*x^2 + b*x + c) + w_sigma, where x ∈ [0:0.99:0.01]
    vector<double> x_data, y_data;      
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma));
    }

    // Start iterations
    int iterations = 100;    // number of iterations
    double cost = 0, lastCost = 0;  // current cost and last cost

    for (int iter = 0; iter < iterations; iter++) {
        Matrix3d H = Matrix3d::Zero();             // Hessian = J^T J in Gauss-Newton
        Vector3d b = Vector3d::Zero();             // bias
        cost = 0;

        for (int i = 0; i < N; i++) {
            double xi = x_data[i], yi = y_data[i];  // ith data point
        
            double error = yi - exp(ae*xi*xi + be*xi + ce); // error
            Vector3d J; 
            J[0] = -xi * xi * exp(ae*xi*xi + be*xi + ce);  // de/da
            J[1] = -xi * exp(ae*xi*xi + be*xi + ce);  // de/db
            J[2] = -exp(ae * xi * xi + be*xi + ce);  // de/dc

            H += J * J.transpose(); // approximated hessian
            b += -error * J;

            cost += error * error;
        }

        // Solve Hx = b using ldlt
        Vector3d dx = H.ldlt().solve(b);

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        // Cost increases (approxiation is not good enough)
        if (iter > 0 && cost > lastCost) {
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // Update a,b,c
        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        lastCost = cost;

        cout << "total cost: " << cost << endl;
    }

    cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
    return 0;
}