sET(TEST_FILES test_triangulation)
FOREACH(src_name ${TEST_FILES})
  add_executable(${src_name} ${src_name}.cpp)
  target_link_libraries(${src_name} ${THIRD_PARTY_LIBS} slamlib)
  add_test(${src_name} ${src_name})
ENDFOREACH(src_name)

