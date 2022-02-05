#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include <sophus/se3.hpp>

const std::string estimation_file_path = "./estimated.txt";
const std::string groundtruth_file_path = "./groundtruth.txt";

bool ReadTrajectory(const std::string &trajectory_file_path, std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> &poses){
    // read file
    std::ifstream reader(trajectory_file_path);
    
    // check if reading is successful
    if (!reader) {
        std::cout << "Unable to read file: " << trajectory_file_path << '\n';
        return false;
    }

    // store trajectory as SE3
    Eigen::Quaterniond q(1,0,0,0);
    Eigen::Vector3d t(0,0,0);
    double time;

    while (!reader.eof()) {
        reader >> time;
        // read t
        reader >> t[0];
        reader >> t[1];
        reader >> t[2];
        // read q
        reader >> q.x();
        reader >> q.y();
        reader >> q.z();
        reader >> q.w();
        Sophus::SE3d SE3 (q,t);
        poses.push_back(SE3);
    }

    reader.close();

    return true;
}

double ComputeRMSE(const std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>& poses_estimated,
                   const std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>& poses_groundtruth) {
    double rmse = 0.0;
    for (int i = 0; i < poses_estimated.size(); ++i) {
        const Sophus::SE3d &pose_groundtruth = poses_groundtruth[i];
        const Sophus::SE3d &pose_estimated = poses_estimated[i];
        // Compute norm of error
        auto error = (poses_groundtruth[i].inverse() * poses_estimated[i]).log().norm();
        rmse += error * error;
    }

    rmse = sqrt(rmse / double(poses_estimated.size()));

    return rmse;
}

int main(int argc, char** argv) {
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses_estimated;
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses_groundtruth;

    // read estimation file
    if (!ReadTrajectory(estimation_file_path, poses_estimated)) {
        std::cout << "Reading estimation file fails...\n";
    } 
    else {
        std::cout << "Reading estimation file succeeds!\n";
    }

    // read groundtruth file
    if (!ReadTrajectory(groundtruth_file_path, poses_groundtruth)) {
        std::cout << "Reading groundtruth file fails...\n";
    } 
    else {
        std::cout << "Reading groundtruth file succeeds!\n";
    }

    std::cout << ComputeRMSE(poses_estimated, poses_groundtruth) << '\n';

    return 0;
}
