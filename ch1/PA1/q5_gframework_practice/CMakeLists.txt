cmake_minimum_required(VERSION 3.1)
project(sayHello)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-std=c++14")

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

#ist(APPEND ${CMAKE_MODULE_PATH} ${PROPROJECT_SOURCE_DIR}/cmake)

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(Glog REQUIRED)
#include_directories(${GLOG_INCLUDE_DIRS})

find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

enable_testing()

add_subdirectory(src)
add_subdirectory(test)
