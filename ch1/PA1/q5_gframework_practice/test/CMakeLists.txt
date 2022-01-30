set(TEST_SOURCES testUseHello)

FOREACH(test_src ${TEST_SOURCES})
    add_executable(${test_src} ${test_src}.cpp)
    target_link_libraries(${test_src} hello)
    add_test(${test_src} ${test_src})
ENDFOREACH (test_src)