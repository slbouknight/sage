cmake_minimum_required(VERSION 3.28)
project(sage LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SAGE_ENABLE_USD "Build the OpenUSD scene importer" OFF)
option(SAGE_ENABLE_CUDA "Build CUDA compute passes" ON)
option(SAGE_BUILD_TESTS "Build unit tests" ON)

include(cmake/SageCompileOptions.cmake)

add_subdirectory(src/core)
add_subdirectory(src/gpu)
add_subdirectory(src/app)
if(SAGE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()