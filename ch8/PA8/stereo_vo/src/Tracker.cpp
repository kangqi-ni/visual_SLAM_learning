#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

#include <opencv2/opencv.hpp>

#include <glog/logging.h>

#include "Tracker.h"
#include "Config.h"
#include "Mappoint.h"
#include "g2o_types.h"

namespace stereo_vo {

Tracker::Tracker():
    state_(NOT_INITIALIZED){
    // Read parameters
    num_features_ = static_cast<size_t>(Config::Read<int>("num_features"));
    min_num_features_init_ = static_cast<size_t>(Config::Read<int>("min_num_features_init"));
    min_num_features_tracking_ = static_cast<size_t>(Config::Read<int>("num_features_tracking"));    
    num_features_tracking_bad_ = static_cast<size_t>(Config::Read<int>("num_features_tracking_bad"));
    num_features_for_new_keyframe_ = static_cast<size_t>(Config::Read<int>("num_features_for_new_keyframe"));

    // Create feature extractor
    gftt_ = cv::GFTTDetector::create(num_features_, 0.01, 20);
}

bool Tracker::AddStereoFrame(Frame::Ptr frame) {
    current_frame_ = frame;

    bool success = false;
    if (state_ == NOT_INITIALIZED) {
        success = StereoInit();
    }
    else if (state_ == GOOD || state_ == BAD) {
        success = Track();
    }
    else {
        LOG(INFO) << "Resetting...\n";
        Reset();
        success = false;
    }

    last_frame_ = current_frame_;
    return success;
}  

bool Tracker::StereoInit() {
    // Extract features in the left image
    size_t num_features = ExtractFeatures();
    LOG(INFO) << "Extract " << num_features << " new features\n";

    // Find correspondences in the right image
    size_t num_correspondences = FindCorrespondences();
    if (num_correspondences < min_num_features_init_) {
        LOG(WARNING) << "Not enough correspondences!\n";
        return false;
    }

    // Initialize the map
    if (MapInit()) {
        LOG(INFO) << "Map is initialized\n";

        state_ = GOOD;
        if (visualizer_) {
            LOG(INFO) << "Starting visualizer...\n";
            visualizer_->AddCurrentFrame(current_frame_);
            visualizer_->UpdateMap();
        }
        return true;
    }
    
    return false;
}

size_t Tracker::ExtractFeatures() {
    // Extract keypoints in the left image
    std::vector<cv::KeyPoint> keypoints;
    gftt_->detect(current_frame_->image_left_, keypoints);

    // Create features
    for (cv::KeyPoint &kp : keypoints) {
        current_frame_->features_left_.push_back(
            Feature::Ptr(new Feature(current_frame_, kp, true)));
    }
    return current_frame_->features_left_.size();
}

size_t Tracker::FindCorrespondences() {
    std::vector<cv::Point2f> pxs_left, pxs_right;
    int num_features = current_frame_->features_left_.size();
    pxs_left.reserve(num_features);
    pxs_right.reserve(num_features);

    // Get keypoints in the left and right images
    for (const Feature::Ptr feature: current_frame_->features_left_) {
        pxs_left.push_back(feature->kp_.pt);

        Mappoint::Ptr mappoint = feature->mappoint_.lock();
        if (mappoint) {
            // Use projected pixel as the initial guess
            Eigen::Vector2f px = camera_right_->world2pixel(mappoint->GetPosition(), current_frame_->GetPose());
            pxs_right.push_back(cv::Point2f(px[0], px[1]));
        }
        else {
            // Same pixel in the left image 
            pxs_right.push_back(feature->kp_.pt);
        }
    }

    // Use optical flow to compute correspondences
    std::vector<uchar> status;
    cv::Mat error;
    cv::calcOpticalFlowPyrLK(
        current_frame_->image_left_, current_frame_->image_right_, pxs_left,
        pxs_right, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // LOG(INFO) << "# candidate correspondences: " << pxs_right.size() << '\n';

    // Compute the number of correspondence inliers
    size_t num_inliers = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        // correspondence
        if (status[i]) {
            Feature::Ptr feature_right (new Feature(current_frame_, cv::KeyPoint(pxs_right[i], 7), false));
            current_frame_->features_right_.push_back(feature_right);
            ++num_inliers;
        }
        // no correspondence
        else {
            current_frame_->features_right_.push_back(nullptr);
        }
    }

    LOG(INFO) << "Find " << num_inliers << " correspondences in the right image\n";
    return num_inliers;
}

bool Tracker::MapInit() {
    std::vector<Sophus::SE3f> poses{camera_left_->GetPose(), camera_right_->GetPose()};
    
    size_t num_initial_points = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        Feature::Ptr feature_left = current_frame_->features_left_[i];
        Feature::Ptr feature_right = current_frame_->features_right_[i];

        // no correspondences
        if (feature_right == nullptr) {
            continue;
        }
        
        // Create mappoints using triangulation
        Eigen::Vector2f px (feature_left->kp_.pt.x, 
                            feature_left->kp_.pt.y);
        Eigen::Vector2f px2 (feature_right->kp_.pt.x, 
                            feature_right->kp_.pt.y);
        std::vector<Eigen::Vector3f> points;
        points.push_back(camera_left_->pixel2cam(px));
        points.push_back(camera_right_->pixel2cam(px2));

        Eigen::Vector3f p_w = Eigen::Vector3f::Zero();
        
        if (Triangulate(poses, points, p_w) && p_w[2] > 0) {
            // Create a mappoint
            Mappoint::Ptr point (new Mappoint(p_w.cast<float>()));
            point->AddObservation(feature_left);
            point->AddObservation(feature_right);
            feature_left->mappoint_ = point;
            feature_right->mappoint_ = point;

            ++num_initial_points;

            // Add the mappoint to the map
            map_->InsertMappoint(point);
        }
    }

    current_frame_->SetKeyframe();
    map_->InsertKeyframe(current_frame_);
    
    optimizer_->UpdateMap();

    // LOG(INFO) << "# initial mappoints: " << num_initial_points << '\n';
    // LOG(INFO) << "# initial keyframes: " << map_->GetAllKeyframes().size() <<'\n';
    return true;
}

bool Tracker::Triangulate(const std::vector<Sophus::SE3f> &poses,
                   const std::vector<Eigen::Vector3f> points, Eigen::Vector3f &pt_world) {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(2 * poses.size(), 4);
    Eigen::Matrix<float, Eigen::Dynamic, 1> b(2 * poses.size());
    b.setZero();

    // Construct Ax = b
    for (size_t i = 0; i < poses.size(); ++i) {
        Eigen::Matrix<float, 3, 4> m = poses[i].matrix3x4();
        A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
        A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
    }
    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

    if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
        return true;
    }

    // Solution is not good enough
    return false;
}

bool Tracker::Track() {
    if (last_frame_) {
        current_frame_->SetPose(relative_motion_ * last_frame_->GetPose());
    }

    size_t num_features_tracked = TrackLastFrame();
    num_tracking_inliers_ = EstimatePose();

    LOG(INFO) << "# inliers: " << num_tracking_inliers_ << '\n';

    if (num_tracking_inliers_ > min_num_features_tracking_) {
        // tracking good
        state_ = GOOD;
    } else if (num_tracking_inliers_ > num_features_tracking_bad_) {
        // tracking bad
        state_ = BAD;
    } else {
        // lost
        state_ = LOST;
    }

    if (visualizer_) {
        visualizer_->AddCurrentFrame(current_frame_);
    }

    InsertKeyframe();

    relative_motion_ = current_frame_->GetPose() * last_frame_->GetPose().inverse();

    return true;
}

size_t Tracker::TrackLastFrame() {
    std::vector<cv::Point2f> pxs1, pxs2;
    int num_features = last_frame_->features_left_.size();
    pxs1.reserve(num_features);
    pxs2.reserve(num_features);

    // LOG(INFO) << "Getting features...\n";

    // Get keypoints in the last and current frames
    size_t i = 0;
    for (const Feature::Ptr feature: last_frame_->features_left_) {
        pxs1.push_back(feature->kp_.pt);

        Mappoint::Ptr mappoint = feature->mappoint_.lock();
        
        if (mappoint) {
            // Use projected pixel as the initial guess
            Eigen::Vector2f px = camera_left_->world2pixel(mappoint->GetPosition(), current_frame_->GetPose());

            pxs2.push_back(cv::Point2f(px[0], px[1]));
        }
        else {
            // Same pixel in the left image 
            pxs2.push_back(feature->kp_.pt);
        }
        ++i;
    }

    // LOG(INFO) << "Using optical flow...\n";

    // Use optical flow to compute correspondences
    std::vector<uchar> status;
    cv::Mat error;
    cv::calcOpticalFlowPyrLK(
        last_frame_->image_left_, current_frame_->image_left_, pxs1,
        pxs2, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // LOG(INFO) << "# candidate correspondences in the current frame: " << pxs2.size() << '\n';

    // Compute the number of correspondence inliers
    size_t num_inliers = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        // correspondence
        if (status[i]) {
            cv::KeyPoint kp(pxs2[i], 7);
            Feature::Ptr feature_current (new Feature(current_frame_, kp, false));
            feature_current->mappoint_ = last_frame_->features_left_[i]->mappoint_;
            current_frame_->features_left_.push_back(feature_current);
            ++num_inliers;
        }
    }

    LOG(INFO) << "Find " << num_inliers << " correspondences in the current image.\n";
    return num_inliers;
}

size_t Tracker::EstimatePose() {
    // Setup g2o optimizer
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg (
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()
        )
    );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // LOG(INFO) << "Initial pose:\n" << current_frame_->GetPose().matrix() << '\n';

    // Set up the pose vertex
    VertexPose *vertex_pose = new VertexPose();
    vertex_pose->setId(0);
    vertex_pose->setEstimate(current_frame_->GetPose().cast<double>());
    optimizer.addVertex(vertex_pose);

    // Set up unary edges
    std::vector<EdgeProjectionPoseOnly*> edges;

    Eigen::Matrix3d K = camera_left_->GetK().cast<double>();

    std::vector<Feature::Ptr> features;

    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        Mappoint::Ptr mappoint = current_frame_->features_left_[i]->mappoint_.lock();
        if (mappoint) {
            features.push_back(current_frame_->features_left_[i]);
            
            EdgeProjectionPoseOnly *edge = new EdgeProjectionPoseOnly(
                (mappoint->GetPosition()).cast<double>(), K
            );
            edge->setId(i);
            edge->setVertex(0, vertex_pose);
            edge->setMeasurement(
                Eigen::Vector2d(current_frame_->features_left_[i]->kp_.pt.x, 
                                current_frame_->features_left_[i]->kp_.pt.y)
            );
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber());
            optimizer.addEdge(edge);

            edges.push_back(edge);
        }
    }

    // LOG(INFO) << "# features: " << features.size() << '\n';
    
    // LOG(INFO) << "Counting outliers...\n";

    // Count the number of inliers
    const double chi2_th = 5.991; // dof = 2 alpha = 0.05
    size_t num_outliers = 0;
    for (int iter = 0; iter < 4; ++iter) {
        vertex_pose->setEstimate(current_frame_->GetPose().cast<double>());
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        num_outliers = 0;

        for (size_t i = 0; i < edges.size(); ++i) {
            EdgeProjectionPoseOnly *e = edges[i];
            if (features[i]->is_outlier_) {
                e->computeError();
            }
            // outlier
            if (e->chi2() > chi2_th) {
                features[i]->is_outlier_ = true;
                e->setLevel(1); // not optimized
                ++num_outliers;
            }
            // inlier 
            else {
                features[i]->is_outlier_ = false;
                e->setLevel(0); // optimized
            }

            if (iter == 2) {
                e->setRobustKernel(nullptr);
            }
        }
    }

    LOG(INFO) << "Outlier/Inlier in pose estimation: " << num_outliers << "/" << features.size() - num_outliers;

    current_frame_->SetPose(vertex_pose->estimate().cast<float>());

    LOG(INFO) << "Current pose:\n" << current_frame_->GetPose().matrix() << '\n';

    for (Feature::Ptr feat: features) {
        if (feat->is_outlier_) {
            feat->mappoint_.reset();
            feat->is_outlier_ = false;  // maybe we can still use it in future
        }
    }

    assert(num_outliers < features.size());
    return features.size() - num_outliers;
}

bool Tracker::InsertKeyframe() {
    if (num_tracking_inliers_ >= num_features_for_new_keyframe_) {
        // Enough inliers
        return false;
    }

    // Add the current frame as the new keyframe
    current_frame_->SetKeyframe();
    map_->InsertKeyframe(current_frame_);
    
    LOG(INFO) << "Setting frame " << current_frame_->GetId() << " as new keyframe " 
              << current_frame_->GetKeyframeId() << '\n';

    // Track in the right image to create more mappoints
    ExtractFeatures();
    FindCorrespondences();
    TriangulateNewPoints();

    optimizer_->UpdateMap();

    if (visualizer_) {
        visualizer_->UpdateMap();
    }

    return true;
}

size_t Tracker::TriangulateNewPoints() {
    std::vector<Sophus::SE3f> poses {camera_left_->GetPose(), camera_right_->GetPose()};
    Sophus::SE3f T_wc = current_frame_->GetPose().inverse();

    size_t num_new_mappoints = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        Feature::Ptr feature_left = current_frame_->features_left_[i];
        Feature::Ptr feature_right = current_frame_->features_right_[i];

        if (feature_left->mappoint_.expired() && feature_right != nullptr) {
            // No matching mappoint but has a correspondence
            // Triangulate to create a mappoint
            std::vector<Eigen::Vector3f> points {
                camera_left_->pixel2cam(Eigen::Vector2f(feature_left->kp_.pt.x, feature_left->kp_.pt.y)),
                camera_right_->pixel2cam(Eigen::Vector2f(feature_right->kp_.pt.x, feature_right->kp_.pt.y))
            };

            Eigen::Vector3f p_w = Eigen::Vector3f::Zero();
            if (Triangulate(poses, points, p_w) && p_w[2] > 0) {
                // Transform to world frame
                p_w = T_wc *  p_w;     

                // Create a mappoint
                Mappoint::Ptr mappoint (new Mappoint(p_w));      

                mappoint->AddObservation(feature_left);
                mappoint->AddObservation(feature_right);

                feature_left->mappoint_ = mappoint;
                feature_right->mappoint_ = mappoint;
                map_->InsertMappoint(mappoint);
                ++num_new_mappoints;
            }
        }
    }

    LOG(INFO) << "# new triangulated mappoints: " << num_new_mappoints << '\n';
    return num_new_mappoints;
}

void Tracker::Reset() {
    // not implemented
}

} // namespace stereo_vo