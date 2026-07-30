find_package(spdlog CONFIG REQUIRED)

add_library(sage_core STATIC
    src/log.cpp
    src/job.cpp
)
add_library(sage::core ALIAS sage_core)

target_include_directories(sage_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(sage_core
    PRIVATE
        sage::compile_options
    PUBLIC
        spdlog::spdlog
)
