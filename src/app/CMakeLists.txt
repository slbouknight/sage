add_executable(sage_app
    src/main.cpp
)
set_target_properties(sage_app PROPERTIES OUTPUT_NAME sage)

target_link_libraries(sage_app PRIVATE
    sage::compile_options
    sage::core
)
