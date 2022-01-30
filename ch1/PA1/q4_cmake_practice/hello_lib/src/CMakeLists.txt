add_library(hello SHARED hello.cpp)
INSTALL(TARGETS hello LIBRARY DESTINATION lib)
INSTALL(FILES ${PROJECT_SOURCE_DIR}/include/hello.h DESTINATION include/hello)