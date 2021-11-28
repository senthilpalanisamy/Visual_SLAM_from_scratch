cmake_minimum_required(VERSION 3.0)
project(orb_manual)

add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")

find_package(OpenCV REQUIRED)

add_executable(orb_manual_exe orb_manual.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(orb_manual_exe ${OpenCV_LIBRARIES})

