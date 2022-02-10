cmake_minimum_required(VERSION 2.8)
project(loop_closure)

set( DBOW3_INCLUDE_DIRS "/usr/local/include/DBoW3")
set( DBOW3_LIBS "/usr/local/lib/libDBoW3.so")

find_package(OpenCV REQUIRED)

include_directories(${DBOW3_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(dictionary_creator feature_detection.cpp)
add_executable(loop_closure loop_closure_test.cpp)

target_link_libraries(dictionary_creator ${DBOW3_LIBS} ${OpenCV_LIBS})
target_link_libraries(loop_closure ${DBOW3_LIBS} ${OpenCV_LIBS})
