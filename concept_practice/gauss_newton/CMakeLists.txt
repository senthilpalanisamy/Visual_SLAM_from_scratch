cmake_minimum_required(VERSION 3.0)
project(gauss_newton)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(EIGEN3_INCLUDE_DIR)
include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(/usr/local/include/eigen3/)

add_executable(gauss gaussNewton.cpp)

target_link_libraries(gauss ${OpenCV_LIBS})
