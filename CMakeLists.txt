project(bovw_place_recognition)
cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(thirdparty/fmt/
                 ${PROJECT_BINARY_DIR}/fmt)

add_subdirectory(thirdparty/googletest/googlemock
                 ${PROJECT_BINARY_DIR}/gtest)
include(CTest)

find_package(OpenCV 4.5.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Python3 COMPONENTS Development NumPy)
include_directories(${Python3_INCLUDE_DIRS} 
  ${Python3_NumPy_INCLUDE_DIRS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

include_directories(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/src)

enable_testing()
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
