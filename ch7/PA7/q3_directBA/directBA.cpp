#include <iostream>

#include <boost/format.hpp>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <Eigen/Core>

#include <sophus/se3.hpp>

#include <opencv2/opencv.hpp>

#include <pangolin/pangolin.h>

using namespace std;

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> VecSE3;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVec3d;

// global variables
string pose_file = "/home/kevin/Documents/vSLAM_learning/ch7/PA7/directBA/poses.txt";
string points_file = "/home/kevin/Documents/vSLAM_learning/ch7/PA7/directBA/points.txt";

// intrinsics
float fx = 277.34;
float fy = 291.402;
float cx = 312.234;
float cy = 239.777;

// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}

// g2o vertex that use sophus::SE3 as pose
class VertexSophus : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexSophus() {}

    ~VertexSophus() {}

    bool read(std::istream &is) {
        return false;
    }

    bool write(std::ostream &os) const {
        return false;
    }

    virtual void setToOriginImpl() {
        _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double *update_) {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> update(update_);
        setEstimate(Sophus::SE3d::exp(update) * estimate());
    }
};

// 16x1 error, which is the error in patch
typedef Eigen::Matrix<double,16,1> Vector16d;
class EdgeDirectProjection : public g2o::BaseBinaryEdge<16, Vector16d, g2o::VertexSBAPointXYZ, VertexSophus> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeDirectProjection(float *color, cv::Mat &target) {
        this->origColor = color;
        this->targetImg = target;
    }

    ~EdgeDirectProjection() {}

    virtual void computeError() override {
        g2o::VertexSBAPointXYZ *point = static_cast<g2o::VertexSBAPointXYZ*>(_vertices[0]);
        VertexSophus *pose = static_cast<VertexSophus*>(_vertices[1]);
        Eigen::Vector3d pc = pose->estimate() * point->estimate();
        float u = pc[0]/pc[2] * fx + cx;
        float v = pc[1]/pc[2] * fy + cy;
        // not in image
        if (u - 2 < 0 || v - 2 < 0 || u + 1 >= targetImg.cols || v+1 >= targetImg.cols) {
            // do not optimize 
            this->setLevel(1);
            for (int i = 0; i < 16; ++i) {
                _error[i] = 0;
            }
        }
        // in image
        else {
            // compute photometric error
            for (int i = -2; i < 2; ++i) {
                for (int j = -2; j < 2; ++j) {
                    int num = 4 * i + j + 10;
                    _error[num] = origColor[num] - GetPixelValue(targetImg, u+i, v+j);
                }
            }
        }
    }

    // Let g2o compute jacobian for you
    virtual void linearizeOplus() override {
        if (level() == 1) {
            _jacobianOplusXi = Eigen::Matrix<double, 16, 3>::Zero();
            _jacobianOplusXj = Eigen::Matrix<double, 16, 6>::Zero();
            return;
        }

        const g2o::VertexSBAPointXYZ *point = static_cast<const g2o::VertexSBAPointXYZ * >(_vertices[0]);
        const VertexSophus *pose = static_cast<const VertexSophus * >(_vertices[1]);
        Eigen::Vector3d Pc = pose->estimate() * point->estimate();

        float x = Pc[0];
        float y = Pc[1];
        float z = Pc[2];
        
        float inv_z = 1.0 / z;
        float inv_z2 = inv_z * inv_z;
        float u = x * inv_z * fx + cx;
        float v = y * inv_z * fy + cy;

        // J_u_q = [fx/Z 0    -fx*X/Z^2]
        //         [0    fy/Z -fy*Y/Z^2]
        Eigen::Matrix<double, 2, 3> J_u_q;
        J_u_q(0, 0) = fx * inv_z;
        J_u_q(0, 1) = 0;
        J_u_q(0, 2) = -fx * x * inv_z2;
        J_u_q(1, 0) = 0;
        J_u_q(1, 1) = fy * inv_z;
        J_u_q(1, 2) = -fy * y * inv_z2;

        // J_q_esp = [I, -q^]
        Eigen::Matrix<double, 3, 6> J_q_esp = Eigen::Matrix<double, 3, 6>::Zero();
        J_q_esp(0, 0) = 1;
        J_q_esp(0, 4) = z;
        J_q_esp(0, 5) = -y;
        J_q_esp(1, 1) = 1;
        J_q_esp(1, 3) = -z;
        J_q_esp(1, 5) = x;
        J_q_esp(2, 2) = 1;
        J_q_esp(2, 3) = y;
        J_q_esp(2, 4) = -x;

        Eigen::Matrix<double, 1, 2> J_I_uv;
        for (int i = -2; i < 2; i++)
            for (int j = -2; j < 2; j++) {
                int num = 4 * i + j + 10;
                // 
                J_I_uv(0, 0) =
                        (GetPixelValue(targetImg, u + i + 1, v + j) - GetPixelValue(targetImg, u + i - 1, v + j)) / 2;
                J_I_uv(0, 1) =
                        (GetPixelValue(targetImg, u + i, v + j + 1) - GetPixelValue(targetImg, u + i, v + j - 1)) / 2;
                _jacobianOplusXi.block<1, 3>(num, 0) = -J_I_uv * J_u_q * pose->estimate().rotationMatrix();
                _jacobianOplusXj.block<1, 6>(num, 0) = -J_I_uv * J_u_q * J_q_esp;
            }
    }

    virtual bool read(istream &in) override {
        return false;
    }

    virtual bool write(ostream &out) const override {
        return false;
    }

private:
    cv::Mat targetImg;  // the target image
    float *origColor = nullptr;   // 16 floats, the color of this point
};

// plot the poses and points
void Draw(const VecSE3 &poses, const VecVec3d &points);

int main(int argc, char **argv) {

    // read poses and points
    VecSE3 poses;
    VecVec3d points;
    ifstream fin(pose_file);

    while (!fin.eof()) {
        double timestamp = 0;
        fin >> timestamp;
        if (timestamp == 0) break;
        double data[7];
        for (auto &d: data) fin >> d;
        poses.push_back(Sophus::SE3d(
                Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                Eigen::Vector3d(data[0], data[1], data[2])
        ));
        if (!fin.good()) break;
    }
    fin.close();

    vector<float *> color;
    fin.open(points_file);
    while (!fin.eof()) {
        double xyz[3] = {0};
        for (int i = 0; i < 3; i++) fin >> xyz[i];
        if (xyz[0] == 0) break;
        points.push_back(Eigen::Vector3d(xyz[0], xyz[1], xyz[2]));
        float *c = new float[16];
        for (int i = 0; i < 16; i++) fin >> c[i];
        color.push_back(c);

        if (fin.good() == false) break;
    }
    fin.close();

    cout << "poses: " << poses.size() << ", points: " << points.size() << endl;

    // read images
    vector<cv::Mat> images;
    boost::format fmt("/home/kevin/Documents/vSLAM_learning/ch7/PA7/directBA/%d.png");
    for (int i = 0; i < 7; i++) {
        images.push_back(cv::imread((fmt % i).str(), 0));
        std::cout << (fmt % i).str() << '\n';
    }

    // build optimization problem
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> DirectBlock; 
    std::unique_ptr<DirectBlock::LinearSolverType> linearSolver (new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>());
    std::unique_ptr<DirectBlock> solver_ptr (new DirectBlock(move(linearSolver)));
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(move(solver_ptr)); // L-M
    
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // add vertices, edges into the graph optimizer
    for (size_t i = 0; i < points.size(); ++i) {
        g2o::VertexSBAPointXYZ *v = new g2o::VertexSBAPointXYZ();
        v->setId(i);
        v->setEstimate(points[i]);
        v->setMarginalized(true);
        optimizer.addVertex(v);
    }

    for (size_t i = 0; i < poses.size(); ++i) {
        VertexSophus *v = new VertexSophus();
        v->setId(points.size() + i);
        v->setEstimate(poses[i]);
        optimizer.addVertex(v);
    }

    for (size_t i = 0; i < poses.size(); ++i) {
        for (size_t j = 0; j < points.size(); ++j) {
            EdgeDirectProjection *e = new EdgeDirectProjection(color[j], images[i]);
            e->setId(i*points.size() + j);
            e->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(j)));
            e->setVertex(1, dynamic_cast<VertexSophus*>(optimizer.vertex(points.size() + i)));
            e->setInformation(Eigen::Matrix<double,16,16>::Identity());
            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber();
            rk->setDelta(1.0); // chi-sqaure threshold for alpha = 0.05 dof = 16
            e->setRobustKernel(rk);
            optimizer.addEdge(e);
        }
    }

    // perform optimization
    optimizer.initializeOptimization(0);
    optimizer.optimize(200);

    // fetch data from the optimizer
    for (size_t i = 0; i < points.size(); ++i) {
        g2o::VertexSBAPointXYZ *v = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i));
        points[i] = v->estimate();
    }

    for (size_t i = 0; i < poses.size(); ++i) {
        VertexSophus *v = static_cast<VertexSophus*>(optimizer.vertex(points.size() + i));
        poses[i] = v->estimate();
    }

    // plot the optimized points and poses
    Draw(poses, points);

    // delete color data
    for (auto &c: color) 
        delete[] c;
    return 0;
}

void Draw(const VecSE3 &poses, const VecVec3d &points) {
    if (poses.empty() || points.empty()) {
        cerr << "parameter is empty!" << endl;
        return;
    }

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // draw poses
        float sz = 0.1;
        int width = 640, height = 480;
        for (auto &Tcw: poses) {
            glPushMatrix();
            Sophus::Matrix4f m = Tcw.inverse().matrix().cast<float>();
            glMultMatrixf((GLfloat *) m.data());
            glColor3f(1, 0, 0);
            glLineWidth(2);
            glBegin(GL_LINES);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glEnd();
            glPopMatrix();
        }

        // points
        glPointSize(2);
        glBegin(GL_POINTS);
        for (size_t i = 0; i < points.size(); i++) {
            glColor3f(0.0, points[i][2]/4, 1.0-points[i][2]/4);
            glVertex3d(points[i][0], points[i][1], points[i][2]);
        }
        glEnd();

        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}

