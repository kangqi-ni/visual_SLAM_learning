#include <opencv2/opencv.hpp>

const std::string input_file = "./fisheye.jpg";

int main(int argc, char **argv) {
    // coefficients 
    double k1 = 0, k2 = 0, k3 = 0, k4 = 0;

    // intrinsics
    double fx = 689.21, fy = 690.48, cx = 1295.56, cy = 942.17;

    cv::Mat image = cv::imread(input_file);
    int rows = image.rows, cols = image.cols;
    cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC3); 

    for (int v = 0; v < rows; v++)
        for (int u = 0; u < cols; u++) {
            // Undistort pixels
            double a = (u - cx) / fx;
            double b = (v - cy) / fy;
            double r = sqrt(a*a + b*b);

            double theta = atan(r);
            double theta_d = theta * (1 + k1*pow(theta,2) + k2*pow(theta,4) + k3*pow(theta,6) + k4*pow(theta,8));

            double x_undistorted = theta_d / r * a;
            double y_undistorted = theta_d / r * b;

            double u_undistorted = fx * x_undistorted + cx;
            double v_undistorted = fy * y_undistorted + cy;

            // Interpolation
            if (u_undistorted >= 0 && v_undistorted >= 0 && u_undistorted < cols &&
                v_undistorted < rows) {
                image_undistort.at<cv::Vec3b>(v, u) =
                    image.at<cv::Vec3b>((int)v_undistorted, (int)u_undistorted);
            } else {
                image_undistort.at<cv::Vec3b>(v, u) = 0;
            }
        }

    // Display undistorted image
    cv::imshow("image undistorted", image_undistort);
    cv::imwrite("fisheye_undist.jpg", image_undistort);
    cv::waitKey();

    return 0;
}
