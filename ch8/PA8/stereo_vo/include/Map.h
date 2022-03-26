#ifndef MAP_H
#define MAP_H

#include <memory>
#include <unordered_map>
#include <mutex>

#include <Eigen/Core>

#include "Frame.h"
#include "Mappoint.h"

namespace stereo_vo {

class Map {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<Map> Ptr;
    typedef std::unordered_map<size_t, Mappoint::Ptr> MappointsType;
    typedef std::unordered_map<size_t, Frame::Ptr> KeyframesType;

    Map();

    // Insert a keyframe into the map
    void InsertKeyframe(Frame::Ptr frame);

    // Insert a mappoints into the map
    void InsertMappoint(Mappoint::Ptr mappoint);

    // Remove old keyframes and their data
    void RemoveOldKeyframes();

    MappointsType GetAllMappoints() {
        std::unique_lock<std::mutex> lock(mappoints_mutex_);
        return mappoints_;
    }

    KeyframesType GetAllKeyframes() {
        std::unique_lock<std::mutex> lock(keyframes_mutex_);
        return keyframes_;
    }

    MappointsType GetActiveMappoints() {
        std::unique_lock<std::mutex> lock(mappoints_mutex_);
        return active_mappoints_;
    }

    KeyframesType GetActiveKeyframes() {
        std::unique_lock<std::mutex> lock(keyframes_mutex_);
        return active_keyframes_;
    }

private:
    MappointsType mappoints_; // mappoints
    MappointsType active_mappoints_; // mappoints in the sliding window
    KeyframesType keyframes_; // keyframes
    KeyframesType active_keyframes_; // keyframes in the sliding window

    Frame::Ptr current_frame_; // current frame

    std::mutex mappoints_mutex_; // mutex for getting and modifying mappoints
    std::mutex keyframes_mutex_; // mutex for getting and modifying keyframe

    size_t max_num_active_keyframes_; // max number of keyframes in the sliding window
}; 

} // namespace stereo_vo

#endif // MAP_H