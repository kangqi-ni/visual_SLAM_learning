#include <opencv2/opencv.hpp>

#include <Eigen/Dense>

#include <sophus/se3.hpp>

#include <string>

using namespace cv;
using namespace std;
using namespace Eigen;

const string left_img_path = "./left.png";
const string right_img_path = "./right.png";
const string disparity_img_path = "./disparity.png";

// camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
double baseline = 0.573;

// useful typedefs
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

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

void OpticalFlowDisparitySingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false
);

void OpticalFlowDisparityMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false
);

double AnalyzeDisparityRMSE(
    const cv::Mat &left_img, 
    const cv::Mat &right_img, 
    cv::Mat &img2, 
    const cv::Mat &disparity_img, 
    const vector<KeyPoint> &kp1, 
    const vector<KeyPoint> &kp2, 
    const vector<bool> &success
);

int main(int argc, char **argv) {
    // images, note they are CV_8UC1, not CV_8UC3
    Mat left_img = imread(left_img_path, 0);
    Mat right_img = imread(right_img_path, 0);
    Mat disparity_img = imread(disparity_img_path, 0);

    // key points, using GFTT here.
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    detector->detect(left_img, kp1);

    // use optical flow to compute disparity
    // single level
    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    OpticalFlowDisparitySingleLevel(left_img, right_img, kp1, kp2_single, success_single, false);

    // multi level
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    OpticalFlowDisparityMultiLevel(left_img, right_img, kp1, kp2_multi, success_multi, false);

    // analyze disparity rmse
    cv::Mat img2_single, img2_multi;
    double cost_single = AnalyzeDisparityRMSE(left_img, right_img, img2_single, disparity_img, kp1, kp2_single, success_single);
    double cost_multi = AnalyzeDisparityRMSE(left_img, right_img, img2_multi, disparity_img, kp1, kp2_multi, success_multi);

    cout << "disparity rmse using optical flow single level : " << cost_single << endl;
    cout << "disparity rmse using optical flow multi level : " << cost_multi << endl;

    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::waitKey(0);

    return 0;
}

void OpticalFlowDisparitySingleLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse
) {
    // parameters
    int half_patch_size = 4;
    int iterations = 10;
    bool have_initial = !kp2.empty();

    for (size_t i = 0; i < kp1.size(); i++) {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx,dy need to be estimated
        if (have_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        for (int iter = 0; iter < iterations; iter++) {
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            cost = 0;

            if (kp.pt.x + dx <= half_patch_size || kp.pt.x + dx >= img1.cols - half_patch_size ||
                kp.pt.y + dy <= half_patch_size || kp.pt.y + dy >= img1.rows - half_patch_size) {   // go outside
                succ = false;
                break;
            }

            // compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {

                    double error = GetPixelValue(img1, kp.pt.x+x, kp.pt.y+y) - GetPixelValue(img2, kp.pt.x+x+dx, kp.pt.y+y+dy);
                    Eigen::Vector2d J;  // Jacobian
                    if (inverse == false) {
                        // Forward Jacobian
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                   GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                   GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
                        );
                    } else if(iter == 0) {
                        // Inverse Jacobian
                        // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img1, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                   GetPixelValue(img1, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                            0.5 * (GetPixelValue(img1, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                   GetPixelValue(img1, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
                        );
                    }

                    // compute H, b and set cost;
                    if (inverse == false || iter == 0){
                        H += J * J.transpose();
                    }
                    b += - J* error;
                    cost += error * error;
                }

            // compute update
            Eigen::Vector2d update = H.ldlt().solve(b);

            if (isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > lastCost) {
                //cout << "cost increased: " << cost << ", " << lastCost << endl;
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;
        }

        success.push_back(succ);

        // set kp2
        if (have_initial) {
            kp2[i].pt = kp.pt + Point2f(dx, dy);
        } else {
            KeyPoint tracked = kp;
            tracked.pt += cv::Point2f(dx, dy);
            kp2.push_back(tracked);
        }
    }
}

void OpticalFlowDisparityMultiLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse) {
    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        }
        else {
            Mat img1_temp, img2_temp;
            resize(pyr1[i-1], img1_temp, Size(pyr1[i-1].cols * pyramid_scale, pyr1[i-1].rows * pyramid_scale));
            resize(pyr2[i-1], img2_temp, Size(pyr2[i-1].cols * pyramid_scale, pyr2[i-1].rows * pyramid_scale));
            pyr1.push_back(img1_temp);
            pyr2.push_back(img2_temp);
        }
    }

    // coarse-to-fine LK tracking in pyramids
    vector<KeyPoint> kp1_pyr, kp2_pyr;
    for (const KeyPoint &kp1: kp1) {
        KeyPoint kp1_top = kp1;
        kp1_top.pt *= scales[pyramids-1];
        kp1_pyr.push_back(kp1_top);
        kp2_pyr.push_back(kp1_top);
    }

    for (int i = pyramids-1; i >= 0; i--) {
        success.clear();
        OpticalFlowDisparitySingleLevel(pyr1[i], pyr2[i], kp1_pyr, kp2_pyr, success, inverse);

        if (i > 0){
            for (cv::KeyPoint &kp:kp1_pyr){
                kp.pt /= pyramid_scale;
            }
            for (cv::KeyPoint &kp:kp2_pyr){
                kp.pt /= pyramid_scale;
            }
        }
    }

    // record keypoints in image 2
    for (const cv::KeyPoint &kp:kp2_pyr){
        kp2.push_back(kp);
    }
}

double AnalyzeDisparityRMSE(const cv::Mat &left_img, const cv::Mat &right_img, cv::Mat &img2, const cv::Mat &disparity_img, const vector<KeyPoint> &kp1, const vector<KeyPoint> &kp2, const vector<bool> &success){
    double cost = 0;
    int count = 0;

    // analyze disparity root mean square error
    cv::cvtColor(right_img, img2, COLOR_GRAY2BGR);
    for (int i = 0; i < kp2.size(); i++) {
        if (success[i]) {
            // draw disparity results
            cv::circle(img2, kp2[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2, kp1[i].pt, cv::Point2f(kp2[i].pt.x, kp1[i].pt.y), cv::Scalar(0, 250, 0));

            // estimated disparity
            double disparity_est = kp2[i].pt.x - kp1[i].pt.x;

            // ground truth disparity
            float u = kp1[i].pt.x;
            float v = kp1[i].pt.y;
            double disparity_gt = GetPixelValue(disparity_img, u, v);

            // compute error
            double error = disparity_gt - disparity_est;
            cost += error * error;
            count++;
        }
    }
    cost = sqrt(cost/count);

    return cost;
}