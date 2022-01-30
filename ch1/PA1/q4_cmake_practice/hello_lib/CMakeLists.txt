cmake_minimum_required(VERSION 3.1)
project(hello_lib)

set(CMAKE_BUILD_TYPE Release)

include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(src)