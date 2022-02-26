cmake_minimum_required(VERSION 3.1)
project(bundle_adjustment)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_FLAGS "-std=c++14")

include_directories(/usr/include/eigen3)

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIR})

add_executable(GN-BA GN-BA.cpp)
target_link_libraries(GN-BA ${Sophus_LIBRARIES})