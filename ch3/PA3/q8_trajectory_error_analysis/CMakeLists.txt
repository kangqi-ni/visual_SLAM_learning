cmake_minimum_required(VERSION 3.1)
project(trajectory_error_analysis)

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIR})

include_directories(/usr/include/eigen3)

add_executable(trajectory_rmse trajectory_rmse.cpp)
target_link_libraries(trajectory_rmse ${Sophus_LIBRARIES})