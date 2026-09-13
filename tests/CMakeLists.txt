find_package(Catch2 3 CONFIG REQUIRED)
include(Catch)

add_executable(sage_tests
    test_logging_listener.cpp
    app/test_edit_history.cpp
    app/test_scene_query.cpp
    app/test_viewport_mapping.cpp
    core/test_camera.cpp
    core/test_handle.cpp
    gpu/test_drag_gesture.cpp
    gpu/test_physical_device_selection.cpp
    gpu/test_primitives.cpp
    gpu/test_scene_graph.cpp
)

target_link_libraries(sage_tests PRIVATE
    sage::compile_options
    sage::app
    sage::core
    sage::gpu
    Catch2::Catch2WithMain
)

catch_discover_tests(sage_tests)
