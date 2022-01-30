cmake_minimum_required(VERSION 3.1)
project(eigen_pratice)

set(CMAKE_BUILD_TYPE Release)

include_directories(/usr/include/eigen3)

add_executable(eigen_practice eigen_practice.cpp)
