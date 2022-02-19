cmake_minimum_required(VERSION 3.1)
project(gauss_newton)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

include_directories(/usr/include/eigen3)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(gaussnewton gaussnewton.cpp)
target_link_libraries(gaussnewton ${OpenCV_LIBS})