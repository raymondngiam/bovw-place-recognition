set(TEST_BINARY ${PROJECT_NAME}_test)
add_executable(${TEST_BINARY} 
               test_main.cpp
               test_data.cpp
               test_bow_dictionary.cpp
               test_bow_dictionary_build.cpp
               test_histograms.cpp
               test_tfidf.cpp)
target_link_libraries(${TEST_BINARY} 
                      gtest
                      gtest_main
                      gmock
                      gmock_main
                      serialize
                      convert_dataset
                      kmeans
                      bow_dictionary
                      histogram
                      tfidf
                      ${OpenCV_LIBS}
                     )
                     
add_test(NAME ${TEST_BINARY}
         COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TEST_BINARY})
