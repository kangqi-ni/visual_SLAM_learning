#include <gtest/gtest.h>
#include "hello.h"

TEST(SayHelloTest, SayHelloNoThrow) {
    EXPECT_NO_THROW(sayHello());
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}