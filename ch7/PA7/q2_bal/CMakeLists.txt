cmake_minimum_required(VERSION 3.1)
project(bal)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_FLAGS "-std=c++14 O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(G2O REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(CSparse REQUIRED)

SET(G2O_LIBS g2o_csparse_extension g2o_stuff g2o_core cxsparse)

include_directories(${PROJECT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})

add_library(bal bal.cpp)

add_executable(compute_bal compute_bal.cpp)
target_link_libraries(compute_bal ${G2O_LIBS} bal)
