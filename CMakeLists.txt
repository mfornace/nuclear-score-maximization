cmake_minimum_required(VERSION 3.8)

################################################################################

set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/external/vcpkg/scripts/buildsystems/vcpkg.cmake
  CACHE STRING "Vcpkg toolchain file")

################################################################################

project(nuclear-score-maximization VERSION 0.0.0 LANGUAGES CXX)

option(NSM_RCHOL "Whether or not rchol should be built" ON)
option(NSM_TEST "Whether or not tests should be built" ON)

################################################################################

find_package(armadillo CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS graph)
find_package(TBB CONFIG REQUIRED)

################################################################################

if(NSM_RCHOL)
    message("-- Using rchol")
    add_library(rchol STATIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/external/rchol/c++/sparse.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/external/rchol/c++/util/util.cpp
        # ${CMAKE_CURRENT_SOURCE_DIR}/external/rchol/c++/rchol/find_separator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/external/rchol/c++/rchol/rchol.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/external/rchol/c++/rchol_lap/rchol_lap.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/external/rchol/c++/rchol_lap/spcol.cpp
    )

    target_include_directories(rchol PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external/rchol/c++)
    target_include_directories(rchol PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/external/rchol/c++/util
        ${CMAKE_CURRENT_SOURCE_DIR}/external/rchol/c++/rchol_lap
    )
else()
    message("-- Not using rchol")
    add_library(rchol INTERFACE)
endif()

################################################################################

add_library(nsm INTERFACE)
target_include_directories(nsm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(nsm INTERFACE cxx_std_20)
target_link_libraries(nsm INTERFACE armadillo Boost::graph nlohmann_json::nlohmann_json TBB::tbb rchol)

################################################################################

if(NSM_TEST)
    message("-- Building tests")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/lilwil)
    add_library(nsm-test SHARED test/Test.cc)
    target_link_libraries(nsm-test PUBLIC liblilwil nsm)
    lilwil_module(nsm-test-module test nsm-test)
else()
    message("-- Not building tests")
endif()

################################################################################