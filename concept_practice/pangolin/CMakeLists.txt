include_directories("/usr/include/eigen3")

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
add_executable(plotTrajectory pangolin_practice.cpp)
target_link_libraries(plotTrajectory ${Pangolin_LIBRARIES})
