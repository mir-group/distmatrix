if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/doctest.h)
    FILE(DOWNLOAD https://raw.githubusercontent.com/onqtam/doctest/2.4.0/doctest/doctest.h ${CMAKE_CURRENT_SOURCE_DIR}/external/doctest.h)
endif()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external)

find_package(LAPACK REQUIRED)