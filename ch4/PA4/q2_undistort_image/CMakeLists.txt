cmake_minimum_required(VERSION 3.1)
project(undistort_image)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(undistort_image undistort_image.cpp)
target_link_libraries(undistort_image ${OpenCV_LIBS})