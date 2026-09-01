find_package(imgui CONFIG REQUIRED)

add_executable(sage_app
    src/application.cpp
    src/file_picker.cpp
    src/main.cpp
)
set_target_properties(sage_app PROPERTIES OUTPUT_NAME sage)

# imgui is declared here rather than inherited: the panels live in this target,
# and it only ever compiled because vcpkg's include directory lands on the
# command line wholesale. Linking worked for an unrelated reason -- CMake keeps
# a static library's PRIVATE dependencies as LINK_ONLY interface requirements --
# so the include path was the accidental half.
target_link_libraries(sage_app PRIVATE
    sage::compile_options
    sage::core
    sage::gpu
    imgui::imgui
)

add_dependencies(sage_app sage_shaders)

target_compile_definitions(sage_app PRIVATE
    SAGE_SHADER_DIR="${SAGE_SHADER_BINARY_DIR}"
    SAGE_ASSET_DIR="${SAGE_ASSET_DIR}"
)
