cmake_minimum_required(VERSION 3.0)
project(motion_estimation)

add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++14 -O2 ${SSE_FLAGS} -msse4")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(g2o REQUIRED)
find_package(Ceres REQUIRED)

add_executable(motion_estimation_exe motion_estimation.cpp feature_matching.cpp)

target_include_directories(motion_estimation_exe PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(motion_estimation_exe PRIVATE feature_matching.hpp)
target_include_directories(motion_estimation_exe PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_include_directories(motion_estimation_exe PRIVATE ${Sophus_INCLUDE_DIRS})
target_include_directories(motion_estimation_exe PRIVATE ${g2o_INCLUDE_DIRS})

target_link_libraries(motion_estimation_exe ${OpenCV_LIBRARIES})
target_link_libraries(motion_estimation_exe Sophus::Sophus)
#target_link_libraries(motion_estimation ${G2O_LIBS})
#target_link_libraries(motion_estimation_exe g2o_core g2o_stuff)

target_link_libraries(motion_estimation_exe g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension ${G2O_LIBS}
${CSPARSE_LIBRARY})
target_link_libraries(motion_estimation_exe ${CERES_LIBRARIES})
