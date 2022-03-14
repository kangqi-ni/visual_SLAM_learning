cmake_minimum_required(VERSION 3.1)
project(forward_additive)


set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++17 -O3 ${SSE_FLAGS} -msse4")

include_directories(/usr/include/eigen3)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(TBB REQUIRED)

add_executable(optical_flow optical_flow.cpp)
target_link_libraries(optical_flow ${OpenCV_LIBS} TBB::tbb)



