This repo contains screen shots of testing ORB_SLAM with a mp4 file and a laptop camera.

The following statements have been added to CMakeLists.txt to test ORB_SLAM:

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/self_recorded_data)

add_executable(myslam
self_recorded_data/myslam.cpp)
target_link_libraries(myslam ${PROJECT_NAME})

add_executable(myvideo
self_recorded_data/myvideo.cpp)
target_link_libraries(myvideo ${PROJECT_NAME})
