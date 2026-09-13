find_package(imgui CONFIG REQUIRED)
find_package(imguizmo CONFIG REQUIRED)

# The editor is a library with a thin executable over it, rather than one
# executable, so that its logic can be linked into the test binary. Nothing in
# here was reachable from tests/ while it was an add_executable, which is why
# the undo stack shipped untested.
add_library(sage_app STATIC
    src/application.cpp
    src/renderer.cpp
    src/edit_history.cpp
    src/scene_query.cpp
    src/viewport_mapping.cpp
    src/file_picker.cpp
)
add_library(sage::app ALIAS sage_app)

target_include_directories(sage_app PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# sage::gpu is PUBLIC where the other modules keep it PRIVATE: application.hpp
# names Device, Swapchain and two dozen more by value, so anything compiling
# against this library needs those headers, not just the linked objects.
#
# imgui stays PRIVATE and is declared here rather than inherited: the panels
# live in this target, and it only ever compiled because vcpkg's include
# directory lands on the command line wholesale. Linking worked for an
# unrelated reason -- CMake keeps a static library's PRIVATE dependencies as
# LINK_ONLY interface requirements -- so the include path was the accidental
# half.
target_link_libraries(sage_app
    PRIVATE
        sage::compile_options
        imgui::imgui
        imguizmo::imguizmo
    PUBLIC
        sage::core
        sage::gpu
)

add_dependencies(sage_app sage_shaders)

target_compile_definitions(sage_app PRIVATE
    SAGE_SHADER_DIR="${SAGE_SHADER_BINARY_DIR}"
    SAGE_ASSET_DIR="${SAGE_ASSET_DIR}"
)

# Named sage_editor because sage_app is now the library, matching sage_core and
# sage_gpu. OUTPUT_NAME keeps the binary at build/<preset>/src/app/sage.
add_executable(sage_editor src/main.cpp)
set_target_properties(sage_editor PROPERTIES OUTPUT_NAME sage)

target_link_libraries(sage_editor PRIVATE
    sage::compile_options
    sage::app
)
