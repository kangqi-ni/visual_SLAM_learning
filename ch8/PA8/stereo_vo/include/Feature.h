#pragma once
#ifndef FEATURE_H
#define FEATURE_H

#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace stereo_vo {
// Forward declarations
class Frame;
class Mappoint;

struct Feature {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<Feature> Ptr;
    
    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp, bool is_in_left): 
        frame_(frame), kp_(kp), is_outlier_(false), is_in_left_(is_in_left){}

    std::weak_ptr<Frame> frame_; // frame that contains the feature
    cv::KeyPoint kp_; // kepoints of the feature
    std::weak_ptr<Mappoint> mappoint_; // mappoints that correspondes to the feature

    bool is_outlier_; 
    bool is_in_left_;
};

} // namespace stereo_vo

#endif // FEATURE_H