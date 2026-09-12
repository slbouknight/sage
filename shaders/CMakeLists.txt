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
set(shaders fxaa mesh outline shadow tonemap)

set(spv_outputs)
foreach(shader IN LISTS shaders)
    set(src "${CMAKE_CURRENT_SOURCE_DIR}/${shader}.slang")
    set(spv "${SAGE_SHADER_BINARY_DIR}/${shader}.spv")

    add_custom_command(
        OUTPUT "${spv}"
        # -fvk-use-entrypoint-name: without it slangc emits a lone entry point
        # as "main" and only preserves source names when a file has more than
        # one. shadow.slang has just a vertex stage, so GraphicsPipeline would
        # look up "vertex_main" and find nothing. Setting it for every shader
        # keeps one rule rather than a per-file exception; the shaders that
        # already have two entry points are unaffected.
        COMMAND "${SLANGC_EXECUTABLE}" "${src}" -target spirv -fvk-use-entrypoint-name
                -o "${spv}" -depfile "${spv}.d"
        DEPENDS "${src}"
        DEPFILE "${spv}.d"
        COMMENT "slangc ${shader}.slang -> ${shader}.spv"
        VERBATIM
    )

    list(APPEND spv_outputs "${spv}")
endforeach()

add_custom_target(sage_shaders ALL DEPENDS ${spv_outputs})