#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pangolin/pangolin.h>

#include <sophus/se3.hpp>

#include <vector>
#include <string>
#include <unistd.h>
#include <fstream>
#include <iostream>

typedef std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;

bool ReadTrajectory(const std::string &trajectory_file_path, TrajectoryType &trajectory_e, TrajectoryType &trajectory_g,
                   std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> &points_e,
                   std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> &points_g) {
    std::ifstream fin(trajectory_file_path);
    // Check if reading is successful
    if (!fin) {
        std::cout << "Unable to read file: " << trajectory_file_path << '\n';
        return false;
    }

    // Store trajectory as SE3
    Eigen::Quaterniond q(1,0,0,0);
    Eigen::Vector3d t(0,0,0);
    double time;

    while (!fin.eof()) {
        // pose estimated
        fin >> time >> t[0] >> t[1] >> t[2] >> q.x() >> q.y() >> q.z() >> q.w();
        points_e.push_back(t);
        Sophus::SE3d SE3_e (q,t);
        trajectory_e.push_back(SE3_e);

        // pose ground truth
        fin >> time >> t[0] >> t[1] >> t[2] >> q.x() >> q.y() >> q.z() >> q.w();
        points_g.push_back(t);
        Sophus::SE3d SE3_g (q,t);
        trajectory_g.push_back(SE3_g);
    }

    fin.close();

    return true;
}

void DrawTrajectory(std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses_e, std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses_g) {
    if (poses_e.empty() || poses_g.empty()) {
        std::cerr << "Trajectory is empty!\n";
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
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        for (size_t i = 0; i < poses_e.size() - 1; i++) {
            glColor3f(1 - (float) i / poses_e.size(), 0.0f, (float) i / poses_e.size());
            glBegin(GL_LINES);
            auto p1 = poses_e[i], p2 = poses_e[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        for (size_t i = 0; i < poses_g.size() - 1; i++) {
            glColor3f(1 - (float) i / poses_g.size(), 0.0f, (float) i / poses_g.size());
            glBegin(GL_LINES);
            auto p1 = poses_g[i], p2 = poses_g[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}

void EstimatePoseICP(const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> &pts1,
                     const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> &pts2,
                     Eigen::Matrix3d &R12, Eigen::Vector3d &t12){
    // Compute the centroids
    // p = 1/N * sum(pi)
    // p' = 1/N * sum(pi')
    Eigen::Vector3d p1, p2; 
    int N = pts1.size();
    for (int i = 0; i < N; ++i){
        p1 += pts1[i];
        p2 += pts2[i];
    }

    p1 /= N;
    p2 /= N;

    std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> q1(N), q2(N); 
    // Remove the centroids
    for (int i = 0; i < N; ++i){
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // Compute W = q * q'^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; ++i){
        W += q1[i] * q2[i].transpose();
    }
    std::cout << "W = " << W << '\n';

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV); 
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    std::cout << "U = " << U << "\n";
    std::cout << "V = " << V << "\n";

    // R = U * V^T
    R12 = U * V.transpose(); 
    if (R12.determinant() < 0) {
        R12 = -R12;
    }

    // p - R * p' - t = 0
    // t = p - R * p'
    t12 = p1 - R12 * p2;

    std::cout << "R = " << R12 << '\n';
    std::cout << "t = " << t12 << '\n';
}

int main(int argc, char** argv) {
    const std::string trajectory_file = "./compare.txt";
    
    // Read trajectories
    TrajectoryType trajectory_e;
    TrajectoryType trajectory_g;
    std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> pts_e;
    std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> pts_g;

    ReadTrajectory(trajectory_file, trajectory_e, trajectory_g, pts_e, pts_g);

    // Use ICP to estimate pose
    Eigen::Matrix3d R_eg;
    Eigen::Vector3d t_eg;
    EstimatePoseICP(pts_e, pts_g, R_eg, t_eg);

    // Transform trajectory_g to the frame of trajectory_e 
    Sophus::SE3d T_eg(R_eg, t_eg);

    for (Sophus::SE3d &pose: trajectory_g) {
        pose = T_eg * pose;
    }

    // Draw trajectories
    DrawTrajectory(trajectory_e, trajectory_g);   
}

