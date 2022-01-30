#include <iostream>
#include <ctime>

#include <Eigen/Core>
#include <Eigen/Dense>

#define MATRIX_SIZE 100

int main(int argc, char** argv) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(MATRIX_SIZE,MATRIX_SIZE);
    // std::cout << "100 x 100 matrix: " << A_100_100 << '\n';

    // Solve A * x = b with QR decomposition and Cholesky decomposition
    Eigen::MatrixXd b =  Eigen::MatrixXd::Random(MATRIX_SIZE, 1);
    Eigen::MatrixXd x;

    clock_t time_stt = clock(); 
    x = A.colPivHouseholderQr().solve(b);
    std::cout << "time used in QR decomposition: " << 1000*(clock() - time_stt)/(double)CLOCKS_PER_SEC << "ms\n";
    std::cout << "QR solution: " << x.transpose() << '\n';

    time_stt= clock();
    x = A.ldlt().solve(b);
    std::cout << "time used in Cholesky decomposition: " << 1000*(clock() - time_stt)/(double)CLOCKS_PER_SEC << "ms\n";
    std::cout << "Cholesky solution: " << x.transpose() << '\n';

    time_stt= clock();
    x = A.inverse() * b;
    std::cout << "time used in inverse multiplication: " << 1000*(clock() - time_stt)/(double)CLOCKS_PER_SEC << "ms\n";
    std::cout << "Inverse solution: " << x.transpose() << '\n';
    return 0;
}