cmake_minimum_required(VERSION 3.1)
project(geometry_practice)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -O3")

include_directories(/usr/include/eigen3)

add_executable(geometry_practice geometry_practice.cpp)
