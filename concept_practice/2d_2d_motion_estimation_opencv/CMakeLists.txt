cmake_minimum_required(VERSION 3.0)
project(motion_estimation)

add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")

find_package(OpenCV REQUIRED)

add_executable(motion_estimation_exe motion_estimation.cpp feature_matching.cpp)

target_include_directories(motion_estimation_exe PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(motion_estimation_exe PRIVATE feature_matching.hpp)
target_link_libraries(motion_estimation_exe ${OpenCV_LIBRARIES})
