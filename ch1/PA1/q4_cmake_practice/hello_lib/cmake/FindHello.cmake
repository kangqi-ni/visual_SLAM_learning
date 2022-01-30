FIND_PATH(HELLO_INCLUDE_DIR NAMES hello.h 
  PATHS
  /usr/local/include
)

FIND_LIBRARY(HELLO_LIBRARY NAMES libhello.so
  PATHS 
  /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HELLO DEFAULT_MSG
  HELLO_INCLUDE_DIR HELLO_LIBRARY)