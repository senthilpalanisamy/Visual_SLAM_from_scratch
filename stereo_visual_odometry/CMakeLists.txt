cmake_minimum_required(VERSION 2.8)
project(stereo_slam)

set(CMAKE_BUILD_TYPE debug)
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4.0.0 REQUIRED)
find_package(Sophus REQUIRED) 
find_package(Ceres REQUIRED)
find_package(GTest REQUIRED)
                                                  

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(./include)



SET(THIRD_PARTY_LIBS ${OpenCV_LIBRARIES} 
                     Sophus::Sophus 
                     ${CSPARSE_LIBRARY} 
                     ${CERES_LIBRARIES} 
                     ${GTEST_BOTH_LIBRARIES})
enable_testing()
add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(test)

