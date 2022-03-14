cmake_minimum_required(VERSION 3.1)
project(directBA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_FLAGS "-std=c++14 O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(G2O REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(CSparse REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)

SET(G2O_LIBS g2o_csparse_extension g2o_stuff g2o_core cxsparse g2o_types_sba)

include_directories(${PROJECT_SOURCE_DIR} 
                    ${EIGEN3_INCLUDE_DIR} 
                    ${CSPARSE_INCLUDE_DIR} 
                    ${OpenCV_INCLUDE_DIRS}
                    ${Pangolin_INCLUDE_DIRS})

add_executable(directBA directBA.cpp)
target_link_libraries(directBA ${G2O_LIBS} ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
