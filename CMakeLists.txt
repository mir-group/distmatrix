cmake_minimum_required(VERSION 3.16)
project(distmatrix)

set(CMAKE_CXX_STANDARD 17)

include_directories(src)
include(external/CMakeLists.txt)

add_library(distmatrix src/matrix.cpp src/matrix.h)
target_link_libraries(distmatrix PUBLIC ${LAPACK_LIBRARIES})
message("Found LAPACK: ${LAPACK_LIBRARIES}")
add_subdirectory(tests)