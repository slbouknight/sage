add_executable(sage_app
    src/application.cpp
    src/main.cpp
)
set_target_properties(sage_app PROPERTIES OUTPUT_NAME sage)

target_link_libraries(sage_app PRIVATE
    sage::compile_options
    sage::core
    sage::gpu
)

add_dependencies(sage_app sage_shaders)

target_compile_definitions(sage_app PRIVATE
    SAGE_SHADER_DIR="${SAGE_SHADER_BINARY_DIR}"
)
