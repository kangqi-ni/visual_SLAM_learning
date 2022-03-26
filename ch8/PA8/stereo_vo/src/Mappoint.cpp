#include "Mappoint.h"

namespace stereo_vo {

Mappoint::Mappoint(const Eigen::Vector3f &position): 
    position_(position), is_outlier_(false){
    // Assign mappoint id
    static size_t factory_id = 0;
    id_ = factory_id++;
}

void Mappoint::AddObservation(Feature::Ptr feature) {
    std::unique_lock<std::mutex> lock(obs_mutex_);
    observations_.push_back(feature);
    ++num_observations_;
}

void Mappoint::RemoveObservation(Feature::Ptr feature) {
    std::unique_lock<std::mutex> lock(obs_mutex_);
    for (auto iter = observations_.begin(); iter != observations_.end(); ++iter) {
        if (iter->lock() == feature) {
            observations_.erase(iter);
            feature->mappoint_.reset();
            --num_observations_;
            break;
        }
    }
}

}