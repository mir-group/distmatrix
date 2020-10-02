if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/doctest.h)
    FILE(DOWNLOAD https://raw.githubusercontent.com/onqtam/doctest/2.4.0/doctest/doctest.h ${CMAKE_CURRENT_SOURCE_DIR}/external/doctest.h)
endif ()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external)

include(ExternalProject)

ExternalProject_Add(eigen_dep
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/eigen_build
        SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/eigen_src
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 5328c9be4356cc3d7ab2e359acb1e9ebeeeea631
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
        )


add_executable(test_main main.cpp basics.cpp linalg.cpp)
add_dependencies(test_main eigen_dep)

target_link_libraries(test_main PUBLIC distmatrix)
target_include_directories(test_main PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/eigen_src)