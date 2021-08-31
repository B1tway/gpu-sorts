cmake_minimum_required(VERSION 3.1)

add_subdirectory(libs)

project(bitonic_cl)

set(CMAKE_CXX_STANDARD 11)


find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(WARNING "OpenMP not found!")
endif()

convertIntoHeader(src/cl/bitonic.cl src/cl/bitonic_cl.h bitonic_kernel)
add_executable(bitonic src/main_bitonic.cpp src/cl/bitonic_cl.h)
target_link_libraries(bitonic libclew libgpu libutils)

