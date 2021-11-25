cmake_minimum_required(VERSION 3.0)
project(ceres_practice)


find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})

#include_directories({})

add_executable(ceres_practice ceres_practice.cpp)
target_link_libraries(ceres_practice ${OpenCV_LIBS} ${CERES_LIBRARIES})
