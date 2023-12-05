add_library(slamlib  SHARED dataset.cpp frame.cpp backend.cpp frontend.cpp visual_odometry.cpp slam_utilities.cpp)
target_link_libraries(slamlib ${THIRD_PARTY_LIBS})
