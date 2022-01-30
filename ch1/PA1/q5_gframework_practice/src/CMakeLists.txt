add_library(hello SHARED hello.cpp)
target_link_libraries(hello 
    glog::glog
    ${GFLAGS_LIBRARIES}
    ${GTEST_BOTH_LIBRARIES}
)

add_executable(sayHello useHello.cpp)
target_link_libraries(sayHello hello)