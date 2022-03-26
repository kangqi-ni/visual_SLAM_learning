#include "Camera.h"

namespace stereo_vo {

Camera::Camera(float fx, float fy, float cx, float cy, float baseline, const Sophus::SE3f &pose):
    fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline), pose_(pose) {}

Eigen::Vector3f Camera::pixel2cam(const Eigen::Vector2f &px, float depth) {
    return Eigen::Vector3f(
        depth * (px[0] - cx_) / fx_,
        depth * (px[1] - cy_) / fy_,
        depth
    );
}

Eigen::Vector3f Camera::world2cam(const Eigen::Vector3f &p_w, const Sophus::SE3f T_cw) {
    return pose_ * T_cw * p_w;
}

Eigen::Vector2f Camera::cam2pixel(const Eigen::Vector3f &p_c) {
    return Eigen::Vector2f(fx_ * p_c[0]/p_c[2] + cx_, fy_ * p_c[1]/p_c[2] + cy_);
}   

Eigen::Vector2f Camera::world2pixel(const Eigen::Vector3f &p_w, const Sophus::SE3f T_cw) {
    return cam2pixel(world2cam(p_w, T_cw));
}

} // namespace stereo_vo