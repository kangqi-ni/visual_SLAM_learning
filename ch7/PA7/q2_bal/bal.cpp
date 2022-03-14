#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "bal.h"
#include "random.h"
#include "rotation.h"

typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;

/// Helper functions
// Read from a file pointer
template<typename T>
void FscanfOrDie(FILE *fptr, const char *format, T *value) {
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1)
        std::cerr << "Invalid UW data file.\n";
}

// Calculate the median 
double Median(std::vector<double> &data) {
    int n = data.size();
    std::vector<double>::iterator mid_point = data.begin() + n/2;
    std::nth_element(data.begin(), mid_point, data.end()); // similar to the partition step in quickSort
    return *mid_point;
}

/// Function definitions
BAL::BAL(const std::string &bal_file_path) {
    // Read bal data
    std::FILE *fptr = fopen(bal_file_path.c_str(), "r");

    if (fptr == nullptr) {
        std::cerr << "Cannot open " << bal_file_path << '\n';
        return;
    }

    std::cout << "Reading " << bal_file_path << '\n';

    FscanfOrDie(fptr, "%lu", &num_cameras_);
    FscanfOrDie(fptr, "%lu", &num_points_);
    FscanfOrDie(fptr, "%lu", &num_observations_);

    point_index_ = new size_t[num_observations_];
    camera_index_ = new size_t[num_observations_];
    observations_ = new double[2*num_observations_];

    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];

    std::cout << "Number of cameras: " << num_cameras_ << '\n';
    std::cout << "Number of points: " << num_points_ << '\n';
    std::cout << "Number of observations: " << num_observations_ << '\n';
    std::cout << "Number of parameters: " << num_parameters_ << '\n';

    for (size_t i = 0; i < num_observations_; ++i) {
        FscanfOrDie(fptr, "%lu", camera_index_+i);
        FscanfOrDie(fptr, "%lu", point_index_+i);
        FscanfOrDie(fptr, "%lf", observations_+2*i);
        FscanfOrDie(fptr, "%lf", observations_+2*i+1);
    }

    for (size_t i = 0; i < num_parameters_; ++i) {
        FscanfOrDie(fptr, "%lf", parameters_+i);
    }

    std::cout << "Reading completed\n"; 

    fclose(fptr);
    // for (size_t i = 0; i < num_observations_; ++i) {
    //     std::cout << camera_index_[i] << ' ' << point_index_[i] << ' ' << observations_[2*i] << ' ' <<  observations_[2*i+1] << '\n';
    // }
}

void BAL::WriteToPLYFile(const std::string &filename) const {
    std::ofstream of(filename.c_str());

    of << "ply"
       << '\n' << "format ascii 1.0"
       << '\n' << "element vertex " << num_cameras_ + num_points_
       << '\n' << "property float x"
       << '\n' << "property float y"
       << '\n' << "property float z"
       << '\n' << "property uchar red"
       << '\n' << "property uchar green"
       << '\n' << "property uchar blue"
       << '\n' << "end_header" << std::endl;

    // Export extrinsic data (i.e. camera centers) as green points.
    double angle_axis[3];
    double center[3];
    for (int i = 0; i < num_cameras(); ++i) {
        const double *camera = cameras() + camera_block_size() * i;
        CameraToAngelAxisAndCenter(camera, angle_axis, center);
        of << center[0] << ' ' << center[1] << ' ' << center[2]
           << " 0 255 0" << '\n';
    }

    // Export the structure (i.e. 3D Points) as white points.
    const double *points = parameters_ + camera_block_size() * num_cameras_;
    for (int i = 0; i < num_points(); ++i) {
        const double *point = points + i * point_block_size();
        for (int j = 0; j < point_block_size(); ++j) {
            of << point[j] << ' ';
        }
        of << " 255 255 255\n";
    }
    of.close();
}

void BAL::Normalize() {
    std::vector<double> tmp(num_points_);
    Eigen::Vector3d median;
    double *points = mutable_points();

    // Compute the marginal median (medians of x,y,z)
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < num_points_; ++j) {
            tmp[j] = points[3 * j + i];
        }
        median(i) = Median(tmp);
    }

    // Subtract the marginal median from each point and store the absolute of each result
    for (int i = 0; i < num_points_; ++i) {
        VectorRef point(points + 3 * i, 3);
        tmp[i] = (point - median).lpNorm<1>();
    }

    // Compute median absolute deviation (medain of all the norms)
    const double median_absolute_deviation = Median(tmp);

    // Scale so that the median absolute deviation of the resulting reconstruction is 100
    const double scale = 100.0 / median_absolute_deviation;

    // X = scale * (X - median)
    for (int i = 0; i < num_points_; ++i) {
        VectorRef point(points + 3 * i, 3);
        point = scale * (point - median);
    }

    // Perform the same normalization for camera centers
    double *cameras = mutable_cameras();
    double angle_axis[3];
    double center[3];
    for (int i = 0; i < num_cameras_; ++i) {
        double *camera = cameras + camera_block_size() * i;
        CameraToAngelAxisAndCenter(camera, angle_axis, center);
        // center = scale * (center - median)
        VectorRef(center, 3) = scale * (VectorRef(center, 3) - median);
        AngleAxisAndCenterToCamera(angle_axis, center, camera);
    }
}

void BAL::CameraToAngelAxisAndCenter(const double *camera,
                                            double *angle_axis,
                                            double *center) const {
    // angle axis
    VectorRef angle_axis_ref(angle_axis, 3);
    angle_axis_ref = ConstVectorRef(camera, 3);

    // center
    // c = -R't
    Eigen::VectorXd inverse_rotation = -angle_axis_ref;
    AngleAxisRotatePoint(inverse_rotation.data(),
                         camera + camera_block_size() - 6,
                         center);
    VectorRef(center, 3) *= -1.0;
}

void BAL::AngleAxisAndCenterToCamera(const double *angle_axis,
                                            const double *center,
                                            double *camera) const {
    // rotation
    ConstVectorRef angle_axis_ref(angle_axis, 3);
    VectorRef(camera, 3) = angle_axis_ref;

    // translation
    // t = -R * c
    AngleAxisRotatePoint(angle_axis, center, camera + camera_block_size() - 6);
    VectorRef(camera + camera_block_size() - 6, 3) *= -1.0;
}

void BAL::Perturb(const double rotation_sigma,
                 const double translation_sigma,
                 const double point_sigma) {
    assert(point_sigma >= 0.0);
    assert(rotation_sigma >= 0.0);
    assert(translation_sigma >= 0.0);

    double *point = mutable_points();
    
    // Add noise to points
    if (point_sigma > 0) {
        for (int i = 0; i < num_points_; ++i) {
            for (int j = 0; j < point_block_size(); ++j) {
                point[3*i+j] += point_sigma * RandNormal();
            }
        }
    }

    // Add noise to poses
    for (int i = 0; i < num_cameras_; ++i) {
        double *camera = mutable_cameras() + camera_block_size() * i;

        double angle_axis[3];
        double center[3];
        // Perturb in the rotation of the camera in the angle-axis representation
        CameraToAngelAxisAndCenter(camera, angle_axis, center);
        if (rotation_sigma > 0.0) {
            for (int j = 0; j < 3; ++j) {
                angle_axis[j] += rotation_sigma * RandNormal();
            }
        }
        AngleAxisAndCenterToCamera(angle_axis, center, camera);

        if (translation_sigma > 0.0) {
            for (int j = 0; j < 3; ++j) {
                camera[camera_block_size() - 6 + j] += translation_sigma * RandNormal();
            }
        }
    }
}



// void PerturbPoint3(const double sigma, double *point) {
//     for (int i = 0; i < 3; ++i)
//         point[i] += RandNormal() * sigma;
// }

// void BAL::Perturb(const double rotation_sigma,
//                          const double translation_sigma,
//                          const double point_sigma) {
//     assert(point_sigma >= 0.0);
//     assert(rotation_sigma >= 0.0);
//     assert(translation_sigma >= 0.0);

//     double *points = mutable_points();
//     if (point_sigma > 0) {
//         for (int i = 0; i < num_points_; ++i) {
//             PerturbPoint3(point_sigma, points + 3 * i);
//         }
//     }

//     for (int i = 0; i < num_cameras_; ++i) {
//         double *camera = mutable_cameras() + camera_block_size() * i;

//         double angle_axis[3];
//         double center[3];
//         // Perturb in the rotation of the camera in the angle-axis
//         // representation
//         CameraToAngelAxisAndCenter(camera, angle_axis, center);
//         if (rotation_sigma > 0.0) {
//             PerturbPoint3(rotation_sigma, angle_axis);
//         }
//         AngleAxisAndCenterToCamera(angle_axis, center, camera);

//         if (translation_sigma > 0.0)
//             PerturbPoint3(translation_sigma, camera + camera_block_size() - 6);
//     }
// }
