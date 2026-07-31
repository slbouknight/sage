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

# SPDLOG_ACTIVE_LEVEL strips SPDLOG_LOGGER_* calls below it at compile time,
# and defaults to INFO -- which would silently discard every SAGE_LOG_DEBUG and
# SAGE_LOG_TRACE regardless of the runtime level set in log.cpp. PUBLIC because
# the macros expand in consumers' translation units, not this one.
target_compile_definitions(sage_core PUBLIC
    $<IF:$<CONFIG:Debug>,SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE,SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO>
)
