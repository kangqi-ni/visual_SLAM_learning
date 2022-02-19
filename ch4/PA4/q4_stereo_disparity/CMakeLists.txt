cmake_minimum_required(VERSION 3.1)
project(stereo_disparity)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIR})

add_executable(disparity disparity.cpp)
target_link_libraries(disparity ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
