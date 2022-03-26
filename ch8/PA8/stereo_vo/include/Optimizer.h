#pragma once
#ifndef Optimizer_H
#define Optimizer_H

#include <memory>
#include <thread>
#include <atomic>
#include <condition_variable>

#include <Eigen/Core>

#include "Map.h"
#include "Camera.h"

namespace stereo_vo {

class Optimizer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<Optimizer> Ptr;

    Optimizer();

    ~Optimizer(){}

    // Take new active keyframes and mappoints for optimization
    void UpdateMap();

    // Stop the optimizer
    void Stop();

    // Reset the optimizer 
    void Reset();

    void SetMap(Map::Ptr map) {map_ = map;}

    void SetCameras(Camera::Ptr left, Camera::Ptr right) {
        camera_left_ = left;
        camera_right_ = right;
    }

private:
    // Start the optimizer thread
    void ThreadLoop();

    // Optimizer keyframes and mappoints
    void Optimize(Map::KeyframesType &keyframes, Map::MappointsType &mappoints);

    std::thread optimizer_thread_;
    std::mutex data_mutex_; // mutex for modifying and updating keyframes and mappoints
    std::condition_variable map_update_;
    std::atomic<bool> optimizer_running_;

    Map::Ptr map_; 
    Camera::Ptr camera_left_;
    Camera::Ptr camera_right_;
};

} // namespace stereo_vo

#endif // Optimizer_H