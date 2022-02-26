cmake_minimum_required(VERSION 3.1)
project(align_trajectory_icp)

include_directories(/usr/include/eigen3)

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIR})

add_executable(align_trajectory_icp align_trajectory_icp.cpp)
target_link_libraries(align_trajectory_icp ${Pangolin_LIBRARIES} ${Sophus_LIBRARIES})

