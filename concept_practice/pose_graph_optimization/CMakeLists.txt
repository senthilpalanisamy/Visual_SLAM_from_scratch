cmake_minimum_required(VERSION 2.8)
project(pose_graph_optimization)

# set(CMAKE_CXX_FLAGS "-std=c++14 -O2")
set(CMAKE_CXX_FLAGS "-std=c++14")

find_package(Sophus REQUIRED)
find_package(g2o REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)


find_library(glog_LIB glog)
#add_library(glog::glog SHARED IMPORTED)

include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${G2O_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${Ceres_INCLUDE_DIRS})

add_executable(pose_graph pose_graph_optimization_SE3.cpp)

target_link_libraries(pose_graph Sophus::Sophus)
target_link_libraries(pose_graph g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension cxsparse ${G2O_LIBS}
  ${CSPARSE_LIBRARY} ${CHOLMOD_LIBRARIES})
target_link_libraries(pose_graph ${Ceres_LIBRARIES})
target_link_libraries(pose_graph glog::glog)
