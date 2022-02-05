#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h>

#include <sophus/se3.hpp>

// need pangolin for plotting trajectory
#include <pangolin/pangolin.h>

using namespace std;

// path to trajectory file
string trajectory_file = "./trajectory.txt";

// function for plotting trajectory, don't edit this code
// start point is red and end point is blue
void DrawTrajectory(vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>);

bool ReadTrajectory(const string &trajectory_file_path, vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> &poses);

int main(int argc, char **argv) {

    vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses;

    // read trajectory file
    if (!ReadTrajectory(trajectory_file, poses)) {
        cout << "Reading fails...\n";
    } 
    else {
        cout << "Reading succeeds!\n";
    }

    // draw trajectory in pangolin
    DrawTrajectory(poses);
    return 0;
}

bool ReadTrajectory(const string &trajectory_file_path, vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> &poses){
    // read file
    ifstream reader(trajectory_file_path);
    
    // check if reading is successful
    if (!reader) {
        cout << "Unable to read file: " << trajectory_file_path << '\n';
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

void DrawTrajectory(vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses) {
    if (poses.empty()) {
        cerr << "Trajectory is empty!\n";
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
        for (size_t i = 0; i < poses.size() - 1; i++) {
            glColor3f(1 - (float) i / poses.size(), 0.0f, (float) i / poses.size());
            glBegin(GL_LINES);
            auto p1 = poses[i], p2 = poses[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}
