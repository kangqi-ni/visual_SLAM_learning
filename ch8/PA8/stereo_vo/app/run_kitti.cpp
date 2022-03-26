#include <iostream>
#include <opencv2/core/utility.hpp>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "System.h"

int main(int argc, char** argv){
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    google::InitGoogleLogging(argv[0]);
    google::SetStderrLogging(google::GLOG_INFO);

    const std::string config_file_path = "config/config.yaml";
    stereo_vo::System system (config_file_path);
    system.Init();
    system.Track();
}