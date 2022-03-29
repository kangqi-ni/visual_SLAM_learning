#include <algorithm>
#include <glog/logging.h>

#include "Map.h"

namespace stereo_vo {

Map::Map():max_num_active_keyframes_(7) {}

void Map::InsertKeyframe(Frame::Ptr frame) {
    std::unique_lock<std::mutex> lock(keyframes_mutex_);

    current_frame_ = frame;

    // Add a new keyframe
    if (keyframes_.find(frame->keyframe_id_) == keyframes_.end()) {
        keyframes_.insert(make_pair(frame->keyframe_id_, frame));
        active_keyframes_.insert(make_pair(frame->keyframe_id_, frame));
    }
    // Update an old keyframe
    else {
        keyframes_[frame->keyframe_id_] = frame;
        active_keyframes_[frame->keyframe_id_] = frame;
    }

    // Remove an old keyframe when the sliding window is too large
    if (active_keyframes_.size() > max_num_active_keyframes_){
        RemoveOldKeyframes();
    }
}

void Map::InsertMappoint(Mappoint::Ptr mappoint) {
    std::unique_lock<std::mutex> lock(mappoints_mutex_);

    // Add a new mappoint
    if (mappoints_.find(mappoint->GetId()) == mappoints_.end()){
        mappoints_.insert(std::make_pair(mappoint->GetId(), mappoint));
        active_mappoints_.insert(std::make_pair(mappoint->GetId(), mappoint));
    }
    // update an old mappoint
    else{
        mappoints_[mappoint->GetId()] = mappoint;
        active_mappoints_[mappoint->GetId()] = mappoint;
    }   
}

void Map::RemoveOldKeyframes() {
    if (current_frame_ == nullptr) {
        return;
    }

    // Find the nearest and farthest active keyframe
    float max_dist = 0, min_dist = FLT_MAX;
    size_t max_keyframe_id, min_keyframe_id;

    Sophus::SE3f T_wc = current_frame_->GetPose().inverse();
    for (auto &keyframe: active_keyframes_) {
        // Do not remove current keyframe
        if (keyframe.second == current_frame_) {
            continue;
        }

        // Define distance as the norm of the lie algebra of relative transformation
        // between 2 keyframes
        float dist = (keyframe.second->GetPose() * T_wc).log().norm();
        if (dist > max_dist) {
            max_dist = dist;
            max_keyframe_id = keyframe.first;
        }
        if (dist < min_dist) {
            min_dist = dist;
            min_keyframe_id = keyframe.first;
        }
    }

    const double min_dist_th = 0.2;
    Frame::Ptr frame_to_remove = nullptr;
    // Remove nearest keyframe
    if (min_dist < min_dist_th) {
        frame_to_remove = keyframes_[min_keyframe_id];
    }
    // Remove farthest keyframe
    else {
        frame_to_remove = keyframes_[max_keyframe_id];
    }

    LOG(INFO) << "Remove keyframe " << frame_to_remove->GetKeyframeId() << "\n";

    active_keyframes_.erase(frame_to_remove->GetKeyframeId());

    // Dissociate features with mappoints
    for (Feature::Ptr feature: frame_to_remove->features_left_) {
        Mappoint::Ptr mappoint = feature->mappoint_.lock();
        if (mappoint) {
            mappoint->RemoveObservation(feature);
        }
    }
    for (Feature::Ptr feature: frame_to_remove->features_right_) {
        if (feature == nullptr) {
            continue;
        }
        Mappoint::Ptr mappoint = feature->mappoint_.lock();
        if (mappoint) {
            mappoint->RemoveObservation(feature);
        }
    }

    std::unique_lock<std::mutex> lock(mappoints_mutex_);

    // Remove mappoints
    size_t num_mappoints_removed = 0;
    for (auto i = active_mappoints_.begin(); i != active_mappoints_.end(); ) {
        if (i->second->GetNumObservations() == 0) {
            i = active_mappoints_.erase(i);
            ++num_mappoints_removed;
        }
        else {
            ++i;
        }
    }

    LOG(INFO) << "Remove " << num_mappoints_removed << " active mappoints";
}   

} // namespace stereo_vo 
