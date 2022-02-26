cmake_minimum_required(VERSION 3.1)
project(recover_pose)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

include_directories(/usr/include/eigen3)

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIR})

add_executable(E2Rt E2Rt.cpp)
target_link_libraries(E2Rt ${Sophus_LIBRARIES})
