#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

string image_file = "./test.png";

int main(int argc, char **argv) {

    // Coefficients
    double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
    // Parameters
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

    cv::Mat image = cv::imread(image_file,0);   
    int rows = image.rows, cols = image.cols;
    cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1);  

    for (int v = 0; v < rows; v++)
        for (int u = 0; u < cols; u++) {
            // Project to camera frame
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;

            // Undistort 
            double r2 = x * x + y * y;
            double x_undistorted = x*(1 + k1*r2 + k2*r2*r2) + 2*p1*x*y + p2*(r2 + 2*x*x);
            double y_undistorted = y*(1 + k1*r2 + k2*r2*r2) + p1*(r2 + 2*y*y) + 2*p2*x*y;

            // Project to image frame
            double u_undistorted = fx * x_undistorted + cx;
            double v_undistorted = fy * y_undistorted + cy;

            // Interpolation
            if (u_undistorted >= 0 && v_undistorted >= 0 && u_undistorted < cols && v_undistorted < rows) {
                image_undistort.at<uchar>(v, u) = image.at<uchar>((int) v_undistorted, (int) u_undistorted);
            } else {
                image_undistort.at<uchar>(v, u) = 0;
            }
        }
    
    // Display undistorted image
    cv::imshow("image undistorted", image_undistort);
    cv::waitKey();

    return 0;
}
