#include <fstream>
#include <sstream>

#include <glog/logging.h>

#include <opencv2/opencv.hpp>

#include "System.h"
#include "Config.h"
#include "Frame.h"

namespace stereo_vo {

System::System(const std::string &config_file_path): config_file_path_(config_file_path) {}

bool System::Init() {
    // Set the config file
    if (Config::SetConfigFile(config_file_path_) == false) {
        return false;
    }

    LOG(INFO) << "Initializing system...\n";

    // Load the sequence
    const std::string image_dir = Config::Read<const std::string>("dataset_dir");

    LOG(INFO) << "Loading the sequence...\n";
    if (LoadSequence(image_dir) == false) {
        return false;
    }

    tracker_ = Tracker::Ptr (new Tracker());
    optimizer_ = Optimizer::Ptr (new Optimizer());
    map_ = Map::Ptr (new Map());
    visualizer_ = Visualizer::Ptr (new Visualizer());

    tracker_->SetMap(map_);
    tracker_->SetOptimizer(optimizer_);
    tracker_->SetVisualizer(visualizer_);
    tracker_->SetCameras(camera_left_, camera_right_);

    optimizer_->SetMap(map_);
    optimizer_->SetCameras(camera_left_, camera_right_);    

    visualizer_->SetMap(map_);

    return true;
}

bool System::Track() {
    LOG(INFO) << "VO running";

    // Start visual odometry
    if (!TrackStereo()) {
        LOG(ERROR) << "Tracking fails.\n";
    }

    // Stop visual odometry
    optimizer_->Stop();
    visualizer_->Close();

    LOG(INFO) << "VO exits";
    return true;
}

bool System::TrackStereo() {
    for (size_t i = 0; i < image_files_left_.size(); ++i) {
        // Read left and right images 
        cv::Mat image_left = cv::imread(image_files_left_[i], cv::IMREAD_GRAYSCALE);
        cv::Mat image_right = cv::imread(image_files_right_[i], cv::IMREAD_GRAYSCALE);
        if (image_left.data == nullptr || image_right.data == nullptr){
            LOG(ERROR) << "Cannot find images at index " << i << '\n';
            return false;
        }

        // Resize images to 1/4
        cv::Mat image_left_resized, image_right_resized;
        cv::resize(image_left, image_left_resized, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
        cv::resize(image_right, image_right_resized, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
        
        // Construct a new frame
        Frame::Ptr frame (new Frame(timestamps_[i], image_left_resized, image_right_resized));

        if (frame == nullptr) {
            LOG(ERROR) << "No more frames to be processed.\n";
            return false;
        }

        // Add the new frame
        auto t1 = std::chrono::steady_clock::now();
        tracker_->AddStereoFrame(frame);
        auto t2 = std::chrono::steady_clock::now();
        auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        LOG(INFO) << "Current frame tracking time: " << time_used.count() << " seconds.";
    }

    LOG(INFO) << "All images have been processed!\n";
    return true;
}

bool System::LoadSequence(const std::string &image_dir) {
    // Load intrinsics and extrinsics
    std::ifstream fin(image_dir + "/calib.txt");
    if (!fin) {
        LOG(ERROR) << "Cannot find " << image_dir + "/calib.txt!\n";
        return false;
    }

    for (int i = 0; i < 2; ++i){
        char camera_name[3];
        for (int k = 0;k < 3; ++k){
            fin >> camera_name[k];
        }
        float projection_data[12];
        for (int k = 0; k < 12; ++k){
            fin >> projection_data[k];
        }

        // K 
        Eigen::Matrix3f K;
        K << projection_data[0], projection_data[1], projection_data[2],
            projection_data[4], projection_data[5], projection_data[6],
            projection_data[8], projection_data[9], projection_data[10];

        // baseline
        Eigen::Vector3f t;
        t << projection_data[3], projection_data[7], projection_data[11];
        t = K.inverse() * t;

        // Since the images are later resized to 1/4 of its orignal size 
        // intrinsics needs to be divided by 2
        K = K * 0.5;
        
        // fx fy cx cy baseline extrinsics
        Camera::Ptr new_camera(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2),
                                          t.norm(), Sophus::SE3f(Sophus::SO3f(), t)));
        if (i == 0) {
            camera_left_ = new_camera;
        }
        else if (i == 1) {
            camera_right_ = new_camera;
        }

        LOG(INFO) << "Camera " << i << " extrinsics: " << t.transpose();
    }
    
    fin.close();

    // Load timestamps
    fin.open(image_dir + "/times.txt");
    if (!fin) {
        LOG(ERROR) << "Cannot find " << image_dir + "/times.txt!\n";
        return false; 
    }

    while (!fin.eof()) {
        std::string s;
        getline(fin, s);
        if (s.empty()) {
            continue;
        }
        std::stringstream ss;
        ss << s;
        double timestamp;
        ss >> timestamp;
        timestamps_.push_back(timestamp);
    }

    LOG(INFO) << "# timestamps: " << timestamps_.size() << '\n';
    fin.close();

    // Load image files
    cv::glob(image_dir + "/image_0", image_files_left_);
    cv::glob(image_dir + "/image_1", image_files_right_);

    LOG(INFO) << "# left images: " << image_files_left_.size() << '\n'; 
    LOG(INFO) << "# right images: " << image_files_right_.size() << '\n';

    return true;
}

} // namespace stereo_vo
