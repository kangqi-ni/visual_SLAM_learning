#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <sophus/se3.hpp>

#include <Eigen/Core>

#include <vector>
#include <string>
#include <boost/format.hpp>
#include <execution>

#include <pangolin/pangolin.h>

using namespace std;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// Camera intrinsics
// 内参
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// 基线
double baseline = 0.573;
// paths
string left_file = "./left.png";
string disparity_file = "./disparity.png";
boost::format fmt_others("./%06d.png");    // other files

// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21
);

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21
);

// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}

template <typename FuncT>
void evaluate_and_call(FuncT func, const std::string &func_name,
                       int times) {
  double total_time = 0;
  for (int i = 0; i < times; ++i) {
    auto t1 = std::chrono::steady_clock::now();
    func();
    auto t2 = std::chrono::steady_clock::now();
    total_time +=
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count() *
        1000;
  }

  std::cout << "方法 " << func_name
            << " 平均调用时间/次数: " << total_time / times << "/" << times
            << " 毫秒." << std::endl;
}

void DirectPoseEstimationSingleLayerMT(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21
);

int main(int argc, char **argv) {

    cv::Mat left_img = cv::imread(left_file, 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);
    
    if (left_img.data == nullptr || disparity_img.data == nullptr){
        cerr << "invalid image file path\n";
        return 1;
    }

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 1000;
    int boarder = 40;
    VecVector2d pixels_ref;
    vector<double> depth_ref;

    // generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; ++i){
        int x = rng.uniform(boarder, left_img.cols - boarder);
        int y = rng.uniform(boarder, left_img.rows - boarder);
        pixels_ref.push_back(Eigen::Vector2d(x,y));

        int disparity = disparity_img.at<uchar> (y,x);
        double depth = fx * baseline / disparity;
        depth_ref.push_back(depth);
    }

    // estimates 01~05.png's pose using this information
    
    Sophus::SE3d T_cur_ref;

    // single layer
    for (int i = 1; i < 6; i++) {  // 1~10
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);    // first you need to test single layer
    }

    // multi layer
    T_cur_ref = Sophus::SE3d();
    for (int i = 1; i < 6; i++) {  // 1~10
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
    }

    // // evaluate parallel programming
    cv::Mat img = cv::imread((fmt_others % 1).str(), 0);
    T_cur_ref = Sophus::SE3d();
    evaluate_and_call([&]() { DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);  },
                        "optical flow single level", 1);
    T_cur_ref = Sophus::SE3d();
    evaluate_and_call([&]() { DirectPoseEstimationSingleLayerMT(left_img, img, pixels_ref, depth_ref, T_cur_ref);  },
                        "optical flow single level multi-thread", 1);
}

void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21
) {

    // parameters
    int half_patch_size = 4;
    int iterations = 100;

    double cost = 0, lastCost = 0;
    int nGood = 0;  // good projections
    VecVector2d goodProjection;

    for (int iter = 0; iter < iterations; iter++) {
        nGood = 0;
        goodProjection.clear();
        goodProjection.resize(px_ref.size());

        // define Hessian and bias
        Matrix6d H = Matrix6d::Zero();  // 6x6 Hessian
        Vector6d b = Vector6d::Zero();  // 6x1 bias

        for (size_t i = 0; i < px_ref.size(); i++) {

            // compute the projection in the second image
            Eigen::Vector3d point_ref = depth_ref[i] * Eigen::Vector3d((px_ref[i](0) - cx)/fx, (px_ref[i](1) - cy)/fy, 1);
            Eigen::Vector3d point_cur = T21 * point_ref;
            if (point_cur[2] < 0) // depth invalid
                continue;

            float u = fx * point_cur(0)/point_cur(2) + cx; 
            float v = fy * point_cur(1)/point_cur(2) + cy;
            
            goodProjection[i] = Eigen::Vector2d(u, v);

            if (u < half_patch_size || v < half_patch_size || u > img1.cols-half_patch_size || v > img1.rows - half_patch_size)
                continue;
            nGood++;

            // parameters
            double X = point_cur(0), Y = point_cur(1), Z = point_cur(2);
            double Z2 = Z * Z, Z_inv = 1./Z, Z2_inv = Z_inv * Z_inv;

            // and compute error and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {

                    double error = GetPixelValue(img1, px_ref[i](0) + x, px_ref[i](1) + y) - GetPixelValue(img2, u+x, v+y);
 
                    Matrix26d J_pixel_xi;
                    J_pixel_xi(0,0) = fx*Z_inv;
                    J_pixel_xi(0,1) = 0;
                    J_pixel_xi(0,2) = -fx*X*Z2_inv;
                    J_pixel_xi(0,3) = -fx*X*Y*Z2_inv;
                    J_pixel_xi(0,4) = fx + fx*X*X*Z2_inv;
                    J_pixel_xi(0,5) = -fx*Y*Z_inv;

                    J_pixel_xi(1,0) = 0;
                    J_pixel_xi(1,1) = fy * Z_inv;
                    J_pixel_xi(1,2) = -fy*Y*Z2_inv;
                    J_pixel_xi(1,3) = -fy-fy*Y*Y*Z2_inv;
                    J_pixel_xi(1,4) = fy*X*Y*Z2_inv;
                    J_pixel_xi(1,5) = fy*X*Z_inv;

                    Eigen::Vector2d J_image_pixel;
                    J_image_pixel = Eigen::Vector2d(
                        0.5 * (GetPixelValue(img2, u+x+1, v+y) - GetPixelValue(img2, u+x-1, v+y)),
                        0.5 * (GetPixelValue(img2, u+x, v+y+1) - GetPixelValue(img2, u+x, v+y-1))
                    );

                    Vector6d J = -1.0 * (J_image_pixel.transpose() * J_pixel_xi).transpose(); 

                    H += J * J.transpose();
                    b += -error * J;
                    cost += error * error;
                }
        }

        // solve update and put it into estimation
        Vector6d update = H.ldlt().solve(b);
        T21 = Sophus::SE3d::exp(update) * T21;

        cost /= nGood;

        if (isnan(update[0])) {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) {
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        lastCost = cost;
        cout << "cost = " << cost << ", good = " << nGood << endl;
    }
    cout << "good projection: " << nGood << endl;
    cout << "T21 = \n" << T21.matrix() << endl;

    // in order to help you debug, we plot the projected pixels here
    cv::Mat img1_show, img2_show;
    cv::cvtColor(img1, img1_show, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, img2_show, cv::COLOR_GRAY2BGR);
    for (auto &px: px_ref) {
        cv::rectangle(img1_show, cv::Point2f(px[0] - 2, px[1] - 2), cv::Point2f(px[0] + 2, px[1] + 2),
                      cv::Scalar(0, 250, 0));
    }
    for (size_t i = 0; i < px_ref.size(); ++i){
        Eigen::Vector2d p_ref = px_ref[i];
        Eigen::Vector2d p_cur = goodProjection[i];
        if (p_cur(0) > 0 && p_cur(1) > 0 && p_ref(0) > 0 && p_ref(1) > 0){
            cv::rectangle(img2_show, cv::Point2f(p_cur[0] - 2, p_cur[1] - 2), 
                      cv::Point2f(p_cur[0] + 2, p_cur[1] + 2),
                      cv::Scalar(0, 250, 0));
            cv::line(img2_show, cv::Point2f(p_ref(0), p_ref(1)), cv::Point2f(p_cur(0), p_cur(1)), cv::Scalar(0,255,0));
        }
    }

    cv::imshow("reference", img1_show);
    cv::imshow("current", img2_show);
    cv::waitKey(100);
}

void DirectPoseEstimationSingleLayerMT(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21
) {

    // parameters
    int half_patch_size = 4;
    int iterations = 100;

    double cost = 0, lastCost = 0;
    int nGood = 0;  // good projections
    VecVector2d goodProjection;

    for (int iter = 0; iter < iterations; iter++) {
        nGood = 0;
        goodProjection.clear();
        goodProjection.resize(px_ref.size());

        // define Hessian and bias
        Matrix6d H = Matrix6d::Zero();  // 6x6 Hessian
        Vector6d b = Vector6d::Zero();  // 6x1 bias

        vector<int> indexs;
        for (size_t i = 0; i < px_ref.size(); i++)
            indexs.push_back(i);

        std::mutex m;
        std::for_each(std::execution::par_unseq, indexs.begin(),indexs.end(),
                      [&] (auto &i){
                            // compute the projection in the second image
                            std::lock_guard<std::mutex> guard(m);
                            Eigen::Vector3d point_ref = depth_ref[i] * Eigen::Vector3d((px_ref[i](0) - cx)/fx, (px_ref[i](1) - cy)/fy, 1);
                            Eigen::Vector3d point_cur = T21 * point_ref;
                            // depth valid
                            if (point_cur[2] >= 0) {
                                float u = fx * point_cur(0)/point_cur(2) + cx; 
                                float v = fy * point_cur(1)/point_cur(2) + cy;
                                
                                goodProjection[i] = Eigen::Vector2d(u, v);

                                if (u >= half_patch_size && v >= half_patch_size && u <= img1.cols-half_patch_size && v <= img1.rows - half_patch_size) {
                                    nGood++;

                                    // parameters
                                    double X = point_cur(0), Y = point_cur(1), Z = point_cur(2);
                                    double Z2 = Z * Z, Z_inv = 1./Z, Z2_inv = Z_inv * Z_inv;

                                    // and compute error and jacobian
                                    for (int x = -half_patch_size; x < half_patch_size; x++)
                                        for (int y = -half_patch_size; y < half_patch_size; y++) {

                                            double error = GetPixelValue(img1, px_ref[i](0) + x, px_ref[i](1) + y) - GetPixelValue(img2, u+x, v+y);
                            
                                            Matrix26d J_pixel_xi;
                                            J_pixel_xi(0,0) = fx*Z_inv;
                                            J_pixel_xi(0,1) = 0;
                                            J_pixel_xi(0,2) = -fx*X*Z2_inv;
                                            J_pixel_xi(0,3) = -fx*X*Y*Z2_inv;
                                            J_pixel_xi(0,4) = fx + fx*X*X*Z2_inv;
                                            J_pixel_xi(0,5) = -fx*Y*Z_inv;

                                            J_pixel_xi(1,0) = 0;
                                            J_pixel_xi(1,1) = fy * Z_inv;
                                            J_pixel_xi(1,2) = -fy*Y*Z2_inv;
                                            J_pixel_xi(1,3) = -fy-fy*Y*Y*Z2_inv;
                                            J_pixel_xi(1,4) = fy*X*Y*Z2_inv;
                                            J_pixel_xi(1,5) = fy*X*Z_inv;

                                            Eigen::Vector2d J_image_pixel;
                                            J_image_pixel = Eigen::Vector2d(
                                                0.5 * (GetPixelValue(img2, u+x+1, v+y) - GetPixelValue(img2, u+x-1, v+y)),
                                                0.5 * (GetPixelValue(img2, u+x, v+y+1) - GetPixelValue(img2, u+x, v+y-1))
                                            );

                                            Vector6d J = -1.0 * (J_image_pixel.transpose() * J_pixel_xi).transpose(); 

                                            H += J * J.transpose();
                                            b += -error * J;
                                            cost += error * error;
                                        }
                                }
                            } 
                          });

        // solve update and put it into estimation
        Vector6d update = H.ldlt().solve(b);
        T21 = Sophus::SE3d::exp(update) * T21;

        cost /= nGood;

        if (isnan(update[0])) {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) {
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        lastCost = cost;
        cout << "cost = " << cost << ", good = " << nGood << endl;
    }
    cout << "good projection: " << nGood << endl;
    cout << "T21 = \n" << T21.matrix() << endl;

    // in order to help you debug, we plot the projected pixels here
    cv::Mat img1_show, img2_show;
    cv::cvtColor(img1, img1_show, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, img2_show, cv::COLOR_GRAY2BGR);
    for (auto &px: px_ref) {
        cv::rectangle(img1_show, cv::Point2f(px[0] - 2, px[1] - 2), cv::Point2f(px[0] + 2, px[1] + 2),
                      cv::Scalar(0, 250, 0));
    }
    for (size_t i = 0; i < px_ref.size(); ++i){
        Eigen::Vector2d p_ref = px_ref[i];
        Eigen::Vector2d p_cur = goodProjection[i];
        if (p_cur(0) > 0 && p_cur(1) > 0 && p_ref(0) > 0 && p_ref(1) > 0 ){
            cv::rectangle(img2_show, cv::Point2f(p_cur[0] - 2, p_cur[1] - 2), 
                      cv::Point2f(p_cur[0] + 2, p_cur[1] + 2),
                      cv::Scalar(0, 250, 0));
            cv::line(img2_show, cv::Point2f(p_ref(0), p_ref(1)), cv::Point2f(p_cur(0), p_cur(1)), cv::Scalar(0,255,0));
        }
    }

    cv::imshow("reference", img1_show);
    cv::imshow("current", img2_show);
    cv::waitKey(100);
}

void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21
) {

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<cv::Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } 
        else {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values
    for (int level = pyramids - 1; level >= 0; level--) {
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px: px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // scale fx, fy, cx, cy in different pyramid levels
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }

}
