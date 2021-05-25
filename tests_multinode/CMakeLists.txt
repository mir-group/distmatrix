if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/doctest.h)
    FILE(DOWNLOAD https://raw.githubusercontent.com/onqtam/doctest/2.4.0/doctest/doctest.h ${CMAKE_CURRENT_SOURCE_DIR}/external/doctest.h)
endif ()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external)

include(ExternalProject)

add_executable(test_multinode main.cpp linalg.cpp)
add_dependencies(test_multinode eigen_dep)

target_link_libraries(test_multinode PUBLIC distmatrix)
target_include_directories(test_multinode PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/eigen_src)
