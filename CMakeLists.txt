cmake_minimum_required(VERSION 3.16)
project(distmatrix C CXX Fortran)

set(CMAKE_CXX_STANDARD 17)


add_library(distmatrix src/matrix.cpp include/matrix.h src/distmatrix.cpp include/distmatrix.h src/blacs.cpp include/blacs.h)

include(external/CMakeLists.txt)

target_include_directories(distmatrix PUBLIC include)
target_include_directories(distmatrix PUBLIC external)
target_link_libraries(distmatrix PUBLIC ${LAPACK_LIBRARIES} -lgfortran)

option(BUILD_TEST "Build the tests (downloads Eigen)" ON)
if (BUILD_TEST)
    add_subdirectory(tests)
    add_subdirectory(tests_multinode)
endif ()
