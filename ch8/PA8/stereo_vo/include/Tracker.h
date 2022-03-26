#pragma once
#ifndef TRACKER_H
#define TRACKER_H

#include <memory>

#include <Eigen/Core>

#include <opencv2/features2d.hpp>

#include "Frame.h"
#include "Map.h"
#include "Visualizer.h"
#include "Optimizer.h"
#include "Camera.h"

namespace stereo_vo {

class Tracker {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<Tracker> Ptr;
    
    Tracker();

    ~Tracker(){}

    // Add a stereo frame
    bool AddStereoFrame(Frame::Ptr frame);

    void SetMap(Map::Ptr map) {map_ = map;}

    void SetOptimizer(Optimizer::Ptr optimizer) {optimizer_ = optimizer;}

    void SetVisualizer(Visualizer::Ptr visualizer) {visualizer_ = visualizer;}

    void SetCameras(Camera::Ptr camera_left, Camera::Ptr camera_right){
        camera_left_ = camera_left;
        camera_right_ = camera_right;
    }

    enum TrackingState{    
        NOT_INITIALIZED=1,         
        GOOD=2,              
        BAD=3,     
        LOST=4             
    };

private:

    // Initialize stereo tracker 
    bool StereoInit();

    // Extract features from the left image
    size_t ExtractFeatures();

    // Find correspondences from the right image
    size_t FindCorrespondences();

    // Initialize the map
    bool MapInit();

    // Linear triangulation using SVD
    bool Triangulate(const std::vector<Sophus::SE3f> &poses,
                     const std::vector<Eigen::Vector3f> points, Eigen::Vector3f &pt_world);

    // Start tracking image
    bool Track();

    // Extract features and find correspondences in the last frame 
    size_t TrackLastFrame();

    // Estimate pose using pnp
    size_t EstimatePose();

    // Insert a keyframe 
    bool InsertKeyframe();

    // Triangulate more mappoints using the left and right images
    size_t TriangulateNewPoints(); 

    // Reset the tracker
    void Reset();

    size_t num_features_; // number of features to be extracted
    size_t min_num_features_init_; // min number of features for initialization
    size_t min_num_features_tracking_; // min number of features for successful tracking
    size_t num_features_tracking_bad_; // number of features for bad tracking
    size_t num_features_for_new_keyframe_ ; // number of features for new keyframe

    size_t num_tracking_inliers_; // number of inliers during tracking

    TrackingState state_;

    Frame::Ptr current_frame_;
    Frame::Ptr last_frame_;

    cv::Ptr<cv::GFTTDetector> gftt_; // feature extractor

    Camera::Ptr camera_left_;
    Camera::Ptr camera_right_;

    Sophus::SE3f relative_motion_; // transformation from last frame to current frame

    Map::Ptr map_;
    Optimizer::Ptr optimizer_;
    Visualizer::Ptr visualizer_;
};

}

#endif // TRACKER_H