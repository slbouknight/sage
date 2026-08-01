find_program(SLANGC_EXECUTABLE slangc
    HINTS "$ENV{VULKAN_SDK}/bin"
    DOC "Slang compiler, ships with the LunarG Vulkan SDK"
)
if(NOT SLANGC_EXECUTABLE)
    message(FATAL_ERROR
        "slangc not found. Source the Vulkan SDK's setup-env.sh before configuring.")
endif()

file(MAKE_DIRECTORY "${SAGE_SHADER_BINARY_DIR}")

# Listed explicitly, never globbed (see CLAUDE.md).
set(shaders triangle)

set(spv_outputs)
foreach(shader IN LISTS shaders)
    set(src "${CMAKE_CURRENT_SOURCE_DIR}/${shader}.slang")
    set(spv "${SAGE_SHADER_BINARY_DIR}/${shader}.spv")

    add_custom_command(
        OUTPUT "${spv}"
        COMMAND "${SLANGC_EXECUTABLE}" "${src}" -target spirv -o "${spv}" -depfile "${spv}.d"
        DEPENDS "${src}"
        DEPFILE "${spv}.d"
        COMMENT "slangc ${shader}.slang -> ${shader}.spv"
        VERBATIM
    )

    list(APPEND spv_outputs "${spv}")
endforeach()

add_custom_target(sage_shaders ALL DEPENDS ${spv_outputs})