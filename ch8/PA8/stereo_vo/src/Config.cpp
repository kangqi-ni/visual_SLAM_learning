#include <glog/logging.h>

#include "Config.h"

namespace stereo_vo{
    
Config::~Config(){
    if (file_.isOpened()){
        file_.release();
    }
}

bool Config::SetConfigFile(const std::string &config_file_path) {
    // Create file pointer
    if (config_ == nullptr) {
        config_ = std::shared_ptr<Config> (new Config());
    }

    // Open config file
    config_->file_ = cv::FileStorage (config_file_path.c_str(), cv::FileStorage::READ);
    if (config_->file_.isOpened() == false){
        LOG(ERROR) << "Cannot open config file!\n";
        config_->file_.release();
        return false;
    }
    
    LOG(INFO) << "Config file is open!\n";
    return true;
}

std::shared_ptr<Config> Config::config_ = nullptr;

} // namespace stereo_vo