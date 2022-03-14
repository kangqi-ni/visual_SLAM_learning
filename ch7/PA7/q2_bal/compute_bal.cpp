#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>

#include "sophus/se3.hpp"

#include "bal.h"

struct PoseAndIntrinsics {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PoseAndIntrinsics(){}

    // Take data from memory
    explicit PoseAndIntrinsics(double *data) {
        // angle axis to so3
        rotation = Sophus::SO3d::exp(Eigen::Vector3d(data[0], data[1], data[2]));
        // translation
        translation = Eigen::Vector3d(data[3], data[4], data[5]);
        // fx, fy
        focal = data[6];
        // k1, k2
        k1 = data[7];
        k2 = data[8];
    }

    // Store estimates into memory
    void set_to(double *data_addr){
        Eigen::Vector3d r = rotation.log();
        for (int i = 0; i < 3; ++i) 
            data_addr[i] = r[i];
        for (int i = 0; i < 3; ++i) 
            data_addr[i+3] = translation[i];
        data_addr[6] = focal;
        data_addr[7] = k1;
        data_addr[8] = k2;
    }

    Sophus::SO3d rotation;
    Eigen::Vector3d translation;
    double focal = 0;
    double k1 = 0, k2 = 0;
};

class VertexPose: public g2o::BaseVertex<9, PoseAndIntrinsics> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual void setToOriginImpl() override {
        _estimate = PoseAndIntrinsics();
    }

    virtual void oplusImpl(const double *update) override {
        _estimate.rotation = Sophus::SO3d::exp(Eigen::Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
        _estimate.translation += Eigen::Vector3d(update[3], update[4], update[5]);
        _estimate.focal += update[6]; 
        _estimate.k1 += update[7];
        _estimate.k2 += update[8];   
    }   

    virtual bool read(std::istream &in) override {
        return false;
    }

    virtual bool write(std::ostream &out) const override {
        return false;
    }

    // Project a point in world frame to pixel in image
    Eigen::Vector2d project(const Eigen::Vector3d &point) {
        Eigen::Vector3d pc = _estimate.rotation * point + _estimate.translation;
        pc = -pc/pc[2];
        double r2 = pc.squaredNorm();
        double distortion = 1.0 + r2 * (_estimate.k1 + _estimate.k2 * r2);
        return Eigen::Vector2d (_estimate.focal * distortion * pc[0], 
                                _estimate.focal * distortion * pc[1]);
    }
};

class VertexPoint: public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setToOriginImpl() override {
        _estimate = Eigen::Vector3d::Zero();
    }

    virtual void oplusImpl(const double *update) override {
        _estimate += Eigen::Vector3d(update[0], update[1], update[2]);
    }

    virtual bool read(std::istream &in) override {
        return false;
    }

    virtual bool write(std::ostream &out) const override {
        return false;
    }
};

class EdgeProjection: public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexPose, VertexPoint> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeProjection(){}

    virtual void computeError() override{
        VertexPose *pose = static_cast<VertexPose*> (_vertices[0]);
        VertexPoint *point = static_cast<VertexPoint*> (_vertices[1]);
        Eigen::Vector2d reprojection = pose->project(point->estimate());
        _error = reprojection - _measurement;
    }

    virtual bool read(std::istream &in) override{
        return false;
    }

    virtual bool write(std::ostream &os) const override{
        return false;
    }
};

void ComputeBAL(BAL &bal){
    // Get parameters
    const size_t point_block_size = bal.point_block_size();
    const size_t camera_block_size = bal.camera_block_size();
    const size_t num_cameras = bal.num_cameras();
    const size_t num_points = bal.num_points();
    const size_t num_observations = bal.num_observations();

    // Get points to data locations
    double *points = bal.mutable_points();
    double *cameras = bal.mutable_cameras();
    const double *observations = bal.observations();

    typedef g2o::BlockSolver< g2o::BlockSolverTraits<9,3>> Block;
    std::unique_ptr<Block::LinearSolverType> linearSolver( new g2o::LinearSolverCSparse<Block::PoseMatrixType>() );     // linear solver
    std::unique_ptr<Block> solver_ptr ( new  Block( move(linearSolver) ) ); 
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( move(solver_ptr) );   // gradient descent 

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    std::vector<VertexPose*> vertex_poses;
    std::vector<VertexPoint*> vertex_points;

    // Add pose vertices
    for (size_t i = 0; i < num_cameras; ++i) {
        VertexPose *v = new VertexPose();
        double *camera = cameras + i*camera_block_size;
        PoseAndIntrinsics estimate (camera);
        v->setId(i);
        v->setEstimate(estimate);
        optimizer.addVertex(v);
        vertex_poses.push_back(v);
    }

    // Add point vertices
    for (size_t i = 0; i < num_points; ++i) {
        VertexPoint *v = new VertexPoint();
        double *point = points + i*point_block_size;
        Eigen::Vector3d estimate (point[0], point[1], point[2]);
        v->setId(num_cameras + i);
        v->setEstimate(estimate);
        v->setMarginalized(true);
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }

    // Add edges
    for (size_t i = 0; i < num_observations; ++i) {
        EdgeProjection *e = new EdgeProjection();
        const double *obs = observations + 2*i;
        e->setId(i);
        e->setVertex(0, vertex_poses[bal.camera_index()[i]]);
        e->setVertex(1, vertex_points[bal.point_index()[i]]);
        e->setMeasurement(Eigen::Vector2d(obs[0], obs[1]));
        e->setInformation(Eigen::Matrix2d::Identity());
        e->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(e);
    }

    std::cout << "Starting optimization....\n";
    optimizer.initializeOptimization();
    optimizer.optimize(40);

    // Set to bal problem
    // optimized poses
    for (int i = 0; i < num_cameras; ++i){
        double *camera = cameras + camera_block_size*i;
        VertexPose *vertex = vertex_poses[i];
        PoseAndIntrinsics estimate = vertex->estimate();
        estimate.set_to(camera);
    }

    // optimized points
    for (int i = 0; i < num_points; ++i){
        double *point = points + point_block_size*i;
        VertexPoint *vertex = vertex_points[i];
        Eigen::Vector3d estimate = vertex->estimate();
        point[0] = estimate(0);
        point[1] = estimate(1);
        point[2] = estimate(2);
    }
}

int main(int argc, char** argv) {
    const std::string bal_file_path = "./problem-52-64053-pre.txt";
    BAL bal(bal_file_path);
    std::cout << "Normalizing data...\n";
    bal.Normalize();

    std::cout << "Adding Gaussian noises...\n";
    bal.Perturb(0.1, 0.5, 0.5);

    std::cout << "Storing initial point clouds...\n";
    bal.WriteToPLYFile("initial.ply");

    std::cout << "Computinig BAL...\n";
    ComputeBAL(bal);

    std::cout << "Storing final point clouds...\n";
    bal.WriteToPLYFile("final.ply");

    std::cout << "Done.\n";
    return 0;
}

