#include <string>

class BAL{
public:
    // Load bal data
    explicit BAL(const std::string &bal_file_path);

    ~BAL() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }

    // Save results to ply pointcloud
    void WriteToPLYFile(const std::string &filename) const;

    // Normalize camera centers and points
    void Normalize();

    // Add Gaussian noises to poses and points
    void Perturb(const double rotation_sigma,
                 const double translation_sigma,
                 const double point_sigma);

    size_t camera_block_size() const { return 9;}

    size_t point_block_size() const { return 3;}

    size_t num_cameras() const { return num_cameras_; }

    size_t num_points() const { return num_points_; }

    size_t num_observations() const { return num_observations_; }

    size_t num_parameters() const { return num_parameters_; }

    const size_t *point_index() const { return point_index_; }

    const size_t *camera_index() const { return camera_index_; }

    const double *observations() const { return observations_; }

    const double *parameters() const { return parameters_; }

    const double *cameras() const { return parameters_; }

    const double *points() const { return parameters_ + camera_block_size() * num_cameras_; }

    const double *camera_for_observation(size_t i) const {
        return cameras() + camera_index_[i] * camera_block_size();
    }

    const double *point_for_observation(size_t i) const {
        return points() + point_index_[i] * point_block_size();
    }

    double *mutable_cameras() { return parameters_; }

    double *mutable_points() { return parameters_ + camera_block_size() * num_cameras_; }

    double *mutable_camera_for_observation(size_t i) {
        return mutable_cameras() + camera_index_[i] * camera_block_size();
    }

    double *mutable_point_for_observation(size_t i) {
        return mutable_points() + point_index_[i] * point_block_size();
    }
    
private:
    // Use pose to compute angle axis and center
    void CameraToAngelAxisAndCenter(const double *camera,
                                    double *angle_axis,
                                    double *center) const;

    // Use angle axis and center to compute pose
    void AngleAxisAndCenterToCamera(const double *angle_axis,
                                    const double *center,
                                    double *camera) const;

    size_t num_cameras_;
    size_t num_points_;
    size_t num_observations_;
    size_t num_parameters_;

    size_t *point_index_;      
    size_t *camera_index_;    
    double *observations_;
    double *parameters_;
};