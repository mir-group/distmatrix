find_package(LAPACK REQUIRED)
message("Found LAPACK: ${LAPACK_LIBRARIES}")

if (LAPACK_LIBRARIES MATCHES mkl)
    add_compile_definitions(USE_MKL)
else ()
    message("LAPACK does not match mkl, building LAPACKE and CBLAS interfaces.")
    include(ExternalProject)
    ExternalProject_Add(lapack_dep
            BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/lapack_build
            SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/lapack_src
            GIT_REPOSITORY https://github.com/Reference-LAPACK/lapack.git
            INSTALL_COMMAND ""
            UPDATE_COMMAND ""
            CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
            -DLAPACKE=ON
            -DCBLAS=ON
            -DUSE_OPTIMIZED_BLAS=ON
            -DUSE_OPTIMIZED_LAPACK=ON
            )
    add_dependencies(distmatrix lapack_dep)
    target_include_directories(distmatrix PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/lapack_build/include)
    target_link_directories(distmatrix PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/lapack_build/lib)
    target_link_libraries(distmatrix PUBLIC cblas PUBLIC lapacke)
    add_compile_definitions(LAPACK_COMPLEX_CPP)
endif ()

find_package(MPI REQUIRED)
target_link_libraries(distmatrix PUBLIC MPI::MPI_CXX)

set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
find_library(SCALAPACK_LIB NAMES mkl_scalapack_lp64 scalapack scalapack-openmpi PATHS ENV LD_LIBRARY_PATH)
if(${SCALAPACK_LIB} MATCHES NOTFOUND)
    ExternalProject_Add(scalapack_dep
            BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/scalapack_build
            SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/scalapack_src
            GIT_REPOSITORY https://github.com/Reference-ScaLAPACK/scalapack.git
            INSTALL_COMMAND ""
            UPDATE_COMMAND ""
            CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
            )
    set(SCALAPACK_LIB scalapack)
    message(WARNING "No SCALAPACK library found in CMAKE_LIBRARY_PATH, building our own.")
    add_dependencies(distmatrix scalapack_dep)
    link_directories(${CMAKE_CURRENT_BINARY_DIR}/scalapack_build/lib)
else()
    if(${SCALAPACK_LIB} MATCHES mkl)
        if("${MPI_CXX_LIBRARIES}" MATCHES openmpi)
            find_library(BLACS_LIB NAMES mkl_blacs_openmpi_lp64 PATHS ENV LD_LIBRARY_PATH)
        elseif("${MPI_CXX_LIBRARIES}" MATCHES intel)
            find_library(BLACS_LIB NAMES mkl_blacs_intelmpi_lp64 PATHS ENV LD_LIBRARY_PATH)
        else()
            message(FATAL_ERROR "Confused by MPI library when looking for BLACS.")
        endif()
        if(${BLACS_LIB} MATCHES NOTFOUND)
            message(FATAL_ERROR "Found Intel SCALAPACK but not BLACS")
        endif()
        message(STATUS "Found BLACS: ${BLACS_LIB}")
        target_link_libraries(distmatrix PUBLIC ${BLACS_LIB})
    endif()
    message(STATUS "Found SCALAPACK: ${SCALAPACK_LIB}")
endif()
target_link_libraries(distmatrix PUBLIC ${SCALAPACK_LIB})
