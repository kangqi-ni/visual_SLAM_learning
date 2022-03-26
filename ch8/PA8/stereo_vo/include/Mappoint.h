#pragma once
#ifndef Mappoint_H
#define Mappoint_H

#include <memory>
#include <list>
#include <mutex>

#include <Eigen/Core>

#include "Frame.h"

namespace stereo_vo{

class Mappoint {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<Mappoint> Ptr;
    Mappoint(const Eigen::Vector3f &position);

    // Add an observation
    void AddObservation(Feature::Ptr feature);

    // Remove an observation
    void RemoveObservation(Feature::Ptr feature);

    size_t GetId() const {return id_;}

    bool IsOutlier() const {return is_outlier_;}
    void SetOutlier(bool is_outlier) {is_outlier_ = is_outlier;}

    Eigen::Vector3f GetPosition() const {return position_;}
    void SetPosition(const Eigen::Vector3f &position) {position_ = position;}

    std::list<std::weak_ptr<Feature>> GetObservations() const {return observations_;}

    size_t GetNumObservations() const {return num_observations_;}

private:
    size_t id_; // mappoint id
    bool is_outlier_; 
    Eigen::Vector3f position_; // 3d position of mappint in world frame

    size_t num_observations_; // number of features that observe the mappoint
    std::list<std::weak_ptr<Feature>> observations_; // features that observe the mappoint

    std::mutex obs_mutex_; // mutex for getting and modifying observations
};

} // namespace stereo_vo

#endif // Mappoint_H