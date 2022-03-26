#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <memory>
#include <thread>

#include <Eigen/Core>

#include <pangolin/pangolin.h>

#include "Map.h"
#include "Frame.h"
#include "Mappoint.h"

namespace stereo_vo {

class Visualizer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Visualizer> Ptr;

    Visualizer();

    void SetMap(Map::Ptr map) {map_ = map;}

    // Close the visualizer
    void Close();

    // Add the current frame
    void AddCurrentFrame(Frame::Ptr current_frame);

    // Update map in the visualizer when there is a new keyframe
    void UpdateMap();
    
    void Reset();

private:
    // Start a visualizer thread
    void ThreadLoop();
    
    // Draw a frame
    void DrawFrame(Frame::Ptr frame, const float* color);

    // Draw mappoints of a frame
    void DrawMapPoints();

    // Follow the current frame in the visualier
    void FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera);

    // Plot the features in current frame into an image
    cv::Mat PlotFrameImage();

    Frame::Ptr current_frame_ = nullptr; 
    Map::Ptr map_ = nullptr;

    std::thread viewer_thread_;
    bool viewer_running_ = true; 

    std::unordered_map<size_t, Frame::Ptr> keyframes_;
    std::unordered_map<size_t, Mappoint::Ptr> landmarks_;
    bool map_updated_ = false;

    std::mutex viewer_data_mutex_;
};

}

#endif // VISUALIZER_H