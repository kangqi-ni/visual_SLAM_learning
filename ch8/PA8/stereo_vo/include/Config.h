#pragma once
#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <string>

namespace stereo_vo {

class Config {
public:
    ~Config();

    // Set the config file
    static bool SetConfigFile(const std::string &config_file_path);

    // Get parameters from the config file
    template<typename T>
    static T Read(const std::string &key) {return T(Config::config_->file_[key]);}

private:
    Config(){}

    cv::FileStorage file_; // config file
    static std::shared_ptr<Config> config_; 
};

} // namespace stereo_vo


#endif // CONFIG_H