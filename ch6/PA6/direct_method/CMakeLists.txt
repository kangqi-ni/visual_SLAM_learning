cmake_minimum_required(VERSION 3.1)
project(direct_method)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++17 -O3 ${SSE_FLAGS} -msse4")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIR})

include_directories(/usr/include/eigen3)

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIR})

find_package(TBB REQUIRED)

add_executable(direct_method direct_method.cpp)
target_link_libraries(direct_method ${OpenCV_LIBS} ${Sophus_LIBRARIES} ${Pangolin_LIBRARIES} TBB::tbb)