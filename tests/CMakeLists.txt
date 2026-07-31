find_package(Catch2 3 CONFIG REQUIRED)
include(Catch)

add_executable(sage_tests
    core/test_handle.cpp
    gpu/test_physical_device_selection.cpp
)

target_link_libraries(sage_tests PRIVATE
    sage::compile_options
    sage::core
    sage::gpu
    Catch2::Catch2WithMain
)

catch_discover_tests(sage_tests)
