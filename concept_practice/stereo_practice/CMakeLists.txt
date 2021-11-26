cmake_minimum_required(VERSION 3.0)
project(stero_vision)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(stereo stereo_vision.cpp)
target_link_libraries(stereo ${OpenCV_LIBRARIES})
