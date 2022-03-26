#pragma once
#ifndef SYSTEM_H
#define SYSTEM_H

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>

#include "Frame.h"
#include "Camera.h"
#include "Tracker.h"
#include "Optimizer.h"
#include "Map.h"
#include "Visualizer.h"

namespace stereo_vo {

class System {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    System(const std::string &config_file_path);

    // Initilize the system
    bool Init();

    // Start tracking
    bool Track();

private:
    // Track one stereo frame
    bool TrackStereo();

    // Load parameters and images
    bool LoadSequence(const std::string &image_dir);

    std::string config_file_path_; // config file path

    std::vector<double> timestamps_; // timestamps
    std::vector<cv::String> image_files_left_; // image files for left camera
    std::vector<cv::String> image_files_right_; // image files for right camera

    Camera::Ptr camera_left_; // left camera model
    Camera::Ptr camera_right_; // right camera model

    Tracker::Ptr tracker_;
    Optimizer::Ptr optimizer_;
    Map::Ptr map_; 
    Visualizer::Ptr visualizer_;
};

} // namespace stereo_vo

#endif // SYSTEM_H