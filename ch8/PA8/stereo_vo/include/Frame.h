#pragma once
#ifndef FRAME_H
#define FRAME_H
#include <mutex>

#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>

#include "Feature.h"

namespace stereo_vo {

class Frame {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<Frame> Ptr;

    Frame(double timestamp, const cv::Mat &image_left, const cv::Mat &image_right);

    // Set the frame as a keyframe
    void SetKeyframe();

    Sophus::SE3f GetPose() {
        std::unique_lock<std::mutex> lock(pose_mutex_);
        return pose_;
    }
    void SetPose(const Sophus::SE3f pose) {
        std::unique_lock<std::mutex> lock(pose_mutex_);
        pose_ = pose;
    }

    size_t GetId() {return id_;}

    size_t GetKeyframeId() {return keyframe_id_;}

public:
    cv::Mat image_left_; // left image
    cv::Mat image_right_; // right image

    double timestamp_; // timestamp of the images
    size_t id_; // frame id
    size_t keyframe_id_; // keyframe id

    bool is_keyframe_;

    std::vector<Feature::Ptr> features_left_; // features in the left image
    std::vector<Feature::Ptr> features_right_; // features in the right image

    Sophus::SE3f pose_; // left camera pose in the frame

    std::mutex pose_mutex_; // mutex for getting and modifying poses
};

} // namespace stereo_vo

#endif // FRAME_H