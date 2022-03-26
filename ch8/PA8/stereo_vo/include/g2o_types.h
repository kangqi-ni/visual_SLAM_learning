#pragma once
#ifndef G2O_TYPES_H
#define G2O_TYPES_H

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <sophus/se3.hpp>

namespace stereo_vo {

// Camera pose
class VertexPose: public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double *update) override {
        Sophus::Vector6d update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    virtual bool read(std::istream &in) override {
        return false;
    }

    virtual bool write(std::ostream &out) const override {
        return false;
    }
};

// Mappoint
class VertexPoint: public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setToOriginImpl() override {
        _estimate = Eigen::Vector3d::Zero();
    }

    virtual void oplusImpl(const double *update) override {
        _estimate[0] += update[0];
        _estimate[1] += update[1];
        _estimate[2] += update[2];
    }

    virtual bool read(std::istream &in) override {
        return false;
    }

    virtual bool write(std::ostream &out) const override {
        return false;
    }
};

// Pose opitmization
class EdgeProjectionPoseOnly: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeProjectionPoseOnly(const Eigen::Vector3d &p_w, const Eigen::Matrix3d &K):
        p_w_(p_w), K_(K) {}

    virtual void computeError() override {
        Sophus::SE3d pose = static_cast<VertexPose*> (_vertices[0])->estimate();
        Eigen::Vector3d px = K_ * (pose * p_w_);
        _error = _measurement - Eigen::Vector2d(px[0]/px[2], px[1]/px[2]);
    }

    virtual void linearizeOplus() override {
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_cam = T * p_w_;
        double fx = K_(0, 0);
        double fy = K_(1, 1);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Zinv = 1.0 / (Z + 1e-18);
        double Zinv2 = Zinv * Zinv;
        _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
            -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
            fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
            -fy * X * Zinv;
    }

    virtual bool read(std::istream &in) override {
        return false;
    }

    virtual bool write(std::ostream &out) const override {
        return false;
    }

private:
    Eigen::Vector3d p_w_; // 3d position of a mappoint
    Eigen::Matrix3d K_; // camera intrinsics
};

// Bundle adjustment
class EdgeProjection: public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexPose, VertexPoint> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeProjection(const Sophus::SE3d extrinsics, const Eigen::Matrix3d K):
        extrinsics_(extrinsics), K_(K) {}

    virtual void computeError() override {
        Sophus::SE3d pose = static_cast<VertexPose*> (_vertices[0])->estimate();
        Eigen::Vector3d point = static_cast<VertexPoint*>(_vertices[1])->estimate();

        Eigen::Vector3d px =  K_ * (extrinsics_ *(pose * point));
        _error = _measurement - Eigen::Vector2d(px[0]/px[2], px[1]/px[2]);
    }

    virtual void linearizeOplus() override {
        const VertexPose *v0 = static_cast<VertexPose*>(_vertices[0]);
        const VertexPoint *v1 = static_cast<VertexPoint*>(_vertices[1]);
        Sophus::SE3d T = v0->estimate();
        Eigen::Vector3d pw = v1->estimate();
        Eigen::Vector3d pos_cam = extrinsics_ * T * pw;
        double fx = K_(0, 0);
        double fy = K_(1, 1);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Zinv = 1.0 / (Z + 1e-18);
        double Zinv2 = Zinv * Zinv;
        _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
            -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
            fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
            -fy * X * Zinv;

        _jacobianOplusXj = _jacobianOplusXi.block<2, 3>(0, 0) *
                           extrinsics_.rotationMatrix() * T.rotationMatrix();
    }

    virtual bool read(std::istream &in) override {
        return false;
    }

    virtual bool write (std::ostream &out) const override {
        return false;
    } 

private:
    Sophus::SE3d extrinsics_; // camera extrinsics
    Eigen::Matrix3d K_; // camera intrinsincs
};

} // namespace stereo_vo

#endif // G2O_TYPES_H