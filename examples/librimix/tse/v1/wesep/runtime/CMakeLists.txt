
cmake_minimum_required(VERSION 3.14)
project(wesep VERSION 0.1)

option(CXX11_ABI "whether to use CXX11_ABI libtorch" OFF)

set(CMAKE_VERBOSE_MAKEFILE OFF)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
get_filename_component(fc_base "fc_base" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC")

include(libtorch)
include(glog)
include(gflags)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# build all libraries
add_subdirectory(utils)
add_subdirectory(frontend)
add_subdirectory(separate)
add_subdirectory(bin)
