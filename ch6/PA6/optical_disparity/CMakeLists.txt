cmake_minimum_required(VERSION 3.1)
project(optical_disparity)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(/usr/include/eigen3)

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIR})

add_executable(optical_disparity optical_disparity.cpp)
target_link_libraries(optical_disparity ${OpenCV_LIBS} ${Sophus_LIBRARIES})