set(TEST_BINARY ${PROJECT_NAME}_test)
add_executable(${TEST_BINARY} 
               test_main.cpp)
target_link_libraries(${TEST_BINARY} 
                      gtest
                      gtest_main
                      serialize
                      convert_dataset
                      ${OpenCV_LIBS}
                     )
                     
add_test(NAME ${TEST_BINARY}
         COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TEST_BINARY})
