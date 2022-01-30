#include "hello.h"
#include <glog/logging.h>
// #include <gflags/gflags.h>

#include <iostream>

void sayHello() {
    LOG(INFO) << "Hello SLAM!";
}
