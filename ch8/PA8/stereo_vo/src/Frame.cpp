#include "Frame.h"

namespace stereo_vo {

Frame::Frame(double timestamp, const cv::Mat &image_left, const cv::Mat &image_right):
    timestamp_(timestamp), image_left_(image_left), image_right_(image_right), is_keyframe_(false) {
    // Assign the frame id
    static size_t factory_id = 0;
    id_ = factory_id++;
}

void Frame::SetKeyframe() {
    // Assign the keyframe id
    static size_t factory_keyframe_id = 0;
    keyframe_id_ = factory_keyframe_id++;
    is_keyframe_ = true;
}

} // namespace stereo_vo