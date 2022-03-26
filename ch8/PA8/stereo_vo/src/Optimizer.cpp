#include <g2o/core/block_solver.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

#include <glog/logging.h>

#include "Optimizer.h"
#include "g2o_types.h"

namespace stereo_vo {

Optimizer::Optimizer() {
    optimizer_running_.store(true);
    optimizer_thread_ = std::thread(std::bind(&Optimizer::ThreadLoop, this));
}

void Optimizer::UpdateMap() {
    // Notify map update
    std::unique_lock<std::mutex> lock(data_mutex_);
    map_update_.notify_one();
}

void Optimizer::ThreadLoop() {
    LOG(INFO) << "Running optimizer...\n";

    while (optimizer_running_.load()) {
        std::unique_lock<std::mutex> lock(data_mutex_);
        map_update_.wait(lock);

        // Optimize active keyframes and mappoints
        Map::KeyframesType active_keyframes = map_->GetActiveKeyframes();
        Map::MappointsType active_mappoints = map_->GetActiveMappoints();
        Optimize(active_keyframes, active_mappoints);
    }
}

void Optimizer::Optimize(Map::KeyframesType &keyframes, Map::MappointsType &mappoints) {
    LOG(INFO) << "Optimizing keyframes and mappoints...\n";

    // Set up g2o optimization algorithm
    typedef g2o::BlockSolver_6_3 BlockType;
    typedef g2o::LinearSolverCSparse<BlockType::PoseMatrixType> LinearSolverType;
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockType>(
            g2o::make_unique<LinearSolverType>()
        )
    );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    size_t max_keyframe_id = 0;
    // Set up pose vertices
    std::map<size_t, VertexPose*> pose_vertices;
    for (const auto &keyframe_pair: keyframes) {
        size_t keyframe_id = keyframe_pair.first;
        Frame::Ptr keyframe = keyframe_pair.second;

        VertexPose *pose_vertex = new VertexPose();
        pose_vertex->setId(keyframe_id);
        pose_vertex->setEstimate(keyframe->GetPose().cast<double>());
        optimizer.addVertex(pose_vertex);

        pose_vertices.insert({keyframe_id, pose_vertex});

        // Update max keyframe id
        if (keyframe_id > max_keyframe_id) {
            max_keyframe_id = keyframe_id;
        }
    }

    Sophus::SE3d left_extrinsics = camera_left_->GetPose().cast<double>();
    Sophus::SE3d right_extrinsics = camera_right_->GetPose().cast<double>();
    Eigen::Matrix3d K = camera_left_->GetK().cast<double>();

    double chi2_th = 5.991; // chi-square dof = 2 alpha = 0.05

    // Set up point vertices and projection edges
    std::map<size_t, VertexPoint*> mappoint_vertices;
    std::map<EdgeProjection*, Feature::Ptr> edges_and_features;

    size_t index = 0; // index for edges
    for (const auto &mappoint_pair: mappoints) {
        Mappoint::Ptr mappoint = mappoint_pair.second;

        if (mappoint->IsOutlier()) {
            continue;
        }

        size_t mappoint_id = mappoint_pair.first;
        std::list<std::weak_ptr<Feature>> observations = mappoint->GetObservations();

        for (std::weak_ptr<Feature> ob: observations) {
            Feature::Ptr feature = ob.lock();
            if (feature == nullptr) {
                continue;
            } 

            Frame::Ptr frame = feature->frame_.lock();
            // Feature is an outlier and not matched with a frame
            if (feature->is_outlier_ || frame == nullptr) {
                continue;
            }

            EdgeProjection *e = nullptr;
            if (feature->is_in_left_) {
                e = new EdgeProjection(left_extrinsics, K);
            }
            else {
                e = new EdgeProjection(right_extrinsics, K);
            }

            // Add mappoint vertex 
            if (mappoint_vertices.find(mappoint_id) == mappoint_vertices.end()) {
                VertexPoint *mappoint_vertex = new VertexPoint();
                mappoint_vertex->setId(max_keyframe_id + mappoint_id + 1);
                mappoint_vertex->setEstimate(mappoint->GetPosition().cast<double>());
                mappoint_vertex->setMarginalized(true);
                optimizer.addVertex(mappoint_vertex);

                mappoint_vertices.insert({mappoint_id, mappoint_vertex});
            }

            e->setId(index++);
            e->setVertex(0, pose_vertices.at(frame->GetKeyframeId()));
            e->setVertex(1, mappoint_vertices.at(mappoint_id));
            e->setMeasurement(Eigen::Vector2d(feature->kp_.pt.x, feature->kp_.pt.y));
            e->setInformation(Eigen::Matrix2d::Identity());
            
            // Set up robust kernel for the edge
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
            rk->setDelta(chi2_th);
            e->setRobustKernel(rk);

            edges_and_features.insert({e, feature});

            optimizer.addEdge(e);
        }
    }
    
    // Optimize and eliminat outliers
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // Determine the threshold for outliers
    size_t num_outliers, num_inliers;
    for (int iter = 0; iter < 5; ++iter) {
        num_outliers = 0;
        num_inliers = 0; 
        for (auto &edge_and_feature: edges_and_features) {
            if (edge_and_feature.first->chi2() > chi2_th) {
                ++num_outliers;
            }
            else {
                ++num_inliers;
            }
        }

        double inlier_ratio = double(num_inliers) / double(num_outliers);
        if (inlier_ratio > 0.5) {
            break;
        }
        else {
            chi2_th *= 2;
        }
    }

    // Classify outliers
    for (auto &edge_and_feature: edges_and_features) {
        if (edge_and_feature.first->chi2() > chi2_th) {
            edge_and_feature.second->is_outlier_ = true;
            // Remove the observations
            edge_and_feature.second->mappoint_.lock()->RemoveObservation(edge_and_feature.second);
        }
        else {
            edge_and_feature.second->is_outlier_ = false;
        }
    }

    LOG(INFO) << "Outlier/Inlier in optimization: " << num_outliers << '/' << num_inliers << '\n';

    // Set poses and mappoints
    for (auto &pose_vertex: pose_vertices) {
        keyframes.at(pose_vertex.first)->SetPose(pose_vertex.second->estimate().cast<float>());
    }

    for (auto &mappoint_vertex: mappoint_vertices) {
        mappoints.at(mappoint_vertex.first)->SetPosition(mappoint_vertex.second->estimate().cast<float>());
    }
}

void Optimizer::Stop() {
    optimizer_running_.store(false);
    map_update_.notify_one();
    optimizer_thread_.join();
}

} // namespace stereo_vo