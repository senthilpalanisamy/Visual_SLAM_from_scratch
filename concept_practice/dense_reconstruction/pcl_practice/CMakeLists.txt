cmake_minimum_required(VERSION 2.8)

project(pcl)
set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS "-std=c++14 -O2")
set(CMAKE_CXX_FLAGS "-std=c++14")


find_package(PCL REQUIRED)
find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${Eigen_INCLUDE_DIRS})


add_executable(pcl_practice pcl_practice.cpp)

target_link_libraries(pcl_practice ${PCL_LIBRARIES})
target_link_libraries(pcl_practice ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
