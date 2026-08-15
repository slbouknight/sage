find_package(spdlog CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

add_library(sage_core STATIC
    src/camera.cpp
    src/log.cpp
    src/job.cpp
    include/sage/core/math.hpp
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
        glm::glm
)

# SPDLOG_ACTIVE_LEVEL strips SPDLOG_LOGGER_* calls below it at compile time,
# and defaults to INFO -- which would silently discard every SAGE_LOG_DEBUG and
# SAGE_LOG_TRACE regardless of the runtime level set in log.cpp. PUBLIC because
# the macros expand in consumers' translation units, not this one.
target_compile_definitions(sage_core PUBLIC
    $<IF:$<CONFIG:Debug>,SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE,SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO>
    # Vulkan's clip space has z in [0,1]; glm targets OpenGL's [-1,1] by
    # default. This must be identical in every translation unit that includes
    # glm -- differing definitions are an ODR violation that silently yields
    # different matrices per TU. Putting it on the target rather than in a
    # header makes that impossible.
    GLM_FORCE_DEPTH_ZERO_TO_ONE
)
