cmake_minimum_required(VERSION 3.0)
project(undistort)

find_package(OpenCV REQUIRED)

add_executable(undistort_exe undistort_image.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})


target_link_libraries(undistort_exe ${OpenCV_LIBRARIES})
