cmake_minimum_required(VERSION 3.1)
project(sayHello)

set(CMAKE_BUILD_TYPE Release)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Hello REQUIRED)
include_directories(${HELLO_INCLUDE_DIR})

add_executable(sayHello useHello.cpp)
target_link_libraries(sayHello ${HELLO_LIBRARY})
