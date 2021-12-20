cmake_minimum_required(VERSION 3.0)
project(optical_flow)

find_package(OpenCV REQUIRED)
add_executable(optical_flow_exe optical_flow.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(optical_flow_exe ${OpenCV_LIBRARIES})
