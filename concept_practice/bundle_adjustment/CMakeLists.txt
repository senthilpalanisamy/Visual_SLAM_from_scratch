cmake_minimum_required(VERSION 3.0)
project(bundle_adjustment)

find_package(Ceres REQUIRED)

add_executable(ceres_ba bundle_adjustment_ceres.cpp common.cpp)

target_link_libraries(ceres_ba ${CERES_LIBRARIES})
