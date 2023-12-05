add_executable(my_slam main.cpp)
target_link_libraries(my_slam ${THIRD_PARTY_LIBS} slamlib)
