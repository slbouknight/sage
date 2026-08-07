cmake_minimum_required(VERSION 3.28)
project(sage LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SAGE_ENABLE_USD "Build the OpenUSD scene importer" OFF)
option(SAGE_ENABLE_CUDA "Build CUDA compute passes" ON)
option(SAGE_BUILD_TESTS "Build unit tests" ON)

# ccache is a pure build-time speedup, never a requirement -- machines without
# it just build normally. Set here rather than in CMakePresets.json so it
# applies to every preset without hardcoding a path to a tool that may not
# exist. Must precede add_subdirectory: the launcher is baked into targets as
# they are defined.
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "ccache enabled: ${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
else()
    message(STATUS "ccache not found; building without compiler caching")
endif()

include(cmake/SageCompileOptions.cmake)

set(SAGE_SHADER_BINARY_DIR "${CMAKE_BINARY_DIR}/shaders")

add_subdirectory(shaders)
add_subdirectory(src/core)
add_subdirectory(src/gpu)
add_subdirectory(src/app)
if(SAGE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()