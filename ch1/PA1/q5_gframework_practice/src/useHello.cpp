#include "hello.h"
#include <glog/logging.h>
#include <gflags/gflags.h>

DEFINE_int32(print_times, 5, "number of times for output");

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    google::InitGoogleLogging(argv[0]);
    google::SetStderrLogging(google::GLOG_INFO);

    if (argv[1]) {
        FLAGS_print_times = std::stoi(argv[1]);
    }
    
    for (int i = 0; i < FLAGS_print_times; ++i) {
        sayHello();
    }
    return 0;
}
