#pragma once
#ifndef CAMERA_H
#define CAMERA_H

#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <sophus/se3.hpp>

namespace stereo_vo{

class Camera{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<Camera> Ptr;

    Camera(float fx, float fy, float cx, float cy, float baseline, const Sophus::SE3f &pose);

    // Project from image frame to camera frame 
    Eigen::Vector3f pixel2cam(const Eigen::Vector2f &px, float depth = 1.0f);

    // Project from world frame to camera frame
    Eigen::Vector3f world2cam(const Eigen::Vector3f &p_w, const Sophus::SE3f T_cw);

    // Project from camera frame to image frame
    Eigen::Vector2f cam2pixel(const Eigen::Vector3f &p_c);

    // Project from world frame to image frame
    Eigen::Vector2f world2pixel(const Eigen::Vector3f &p_w, const Sophus::SE3f T_cw);

    Sophus::SE3f GetPose() const {return pose_;}

    Eigen::Matrix3f GetK() const {
        Eigen::Matrix3f K;
        K << fx_, 0, cx_, 0, fy_, cy_, 0, 0 ,1;
        return K;
    }

private:
    float fx_, fy_, cx_, cy_, baseline_; // intrinsics
    Sophus::SE3f pose_; // extrinsics
};

} // namespace stereo_vo

#endif // CAMERA_H