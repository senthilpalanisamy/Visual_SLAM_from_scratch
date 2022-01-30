cmake_minimum_required(VERSION 3.11)
project(bundle_adjustment)
set(CMAKE_CXX_FLAGS "-std=c++14") 

find_package(Ceres REQUIRED)
find_package(g2o REQUIRED)
find_package(Sophus REQUIRED)
#find_package(suitesparse REQUIRED)

add_executable(ceres_ba bundle_adjustment_ceres.cpp common.cpp)
add_executable(g2o_ba bundle_adjustment_g2o.cpp common.cpp)

target_include_directories(g2o_ba PRIVATE ${g2o_INCLUDE_DIRS})
target_include_directories(g2o_ba PRIVATE ${Sophus_INCLUDE_DIRS})
target_include_directories(g2o_ba PRIVATE "/usr/include/suitesparse")


target_link_libraries(ceres_ba ${CERES_LIBRARIES})

target_link_libraries(g2o_ba ${CERES_LIBRARIES})
target_link_libraries(g2o_ba g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension cxsparse ${G2O_LIBS}
${CSPARSE_LIBRARY})
target_link_libraries(g2o_ba Sophus::Sophus)
# target_link_libraries(g2o_ba "/usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so")

