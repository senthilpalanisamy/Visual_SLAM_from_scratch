cmake_minimum_required(VERSION 2.8)
project(stereo_slam)

set(CMAKE_BUILD_TYPE debug)
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")

add_executable(slam src/main.cpp src/dataset.cpp src/frame.cpp src/backend.cpp src/frontend.cpp src/visual_odometry.cpp src/slam_utilities.cpp)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4.0.0 REQUIRED)
find_package(Sophus REQUIRED)             
find_package(Ceres REQUIRED)
                                                  

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(./include)

target_link_libraries(slam ${OpenCV_LIBRARIES} Sophus::Sophus ${CSPARSE_LIBRARY} ${CERES_LIBRARIES})
