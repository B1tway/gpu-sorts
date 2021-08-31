add_executable(fp32_v_merge fp32_v_merge.cpp)
target_link_libraries(fp32_v_merge examples_common)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fp32_v_merge DESTINATION bin COMPONENT examples)