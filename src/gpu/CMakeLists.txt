find_package(Vulkan QUIET)
if(NOT Vulkan_FOUND)
    if(DEFINED ENV{VULKAN_SDK})
        message(FATAL_ERROR "Vulkan not found under VULKAN_SDK=$ENV{VULKAN_SDK}")
    endif()
    message(FATAL_ERROR
        "Vulkan not found. Source the LunarG SDK's setup-env.sh before configuring, e.g.\n"
        "    source ~/vulkansdk/<version>/setup-env.sh")
endif()
find_package(fastgltf CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
# stb ships no imported target -- the port exports only Stb_INCLUDE_DIR, so
# this is an include directory rather than a target_link_libraries entry.
find_package(Stb REQUIRED)

add_library(sage_gpu STATIC
    src/allocator.cpp
    src/bindless_set.cpp
    src/buffer.cpp
    src/depth_buffer.cpp
    src/device.cpp
    src/frame_pacer.cpp
    src/geometry_registry.cpp
    src/gltf_loader.cpp
    src/imgui_layer.cpp
    src/instance.cpp
    src/material_registry.cpp
    src/physical_device.cpp
    src/pipeline.cpp
    src/pipeline_cache.cpp
    src/sampler.cpp
    src/scene.cpp
    src/shader_module.cpp
    src/surface.cpp
    src/swapchain.cpp
    src/texture.cpp
    src/texture_registry.cpp
    src/uploader.cpp
    src/vk_check.cpp
    src/window.cpp
)
add_library(sage::gpu ALIAS sage_gpu)

target_include_directories(sage_gpu PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# PRIVATE: stb_image.h is included by texture.cpp alone, which is also the
# single translation unit defining STB_IMAGE_IMPLEMENTATION.
# SYSTEM: stb is public-domain C, and its implementation trips -Wold-style-cast,
# -Wconversion and friends by the dozen. -isystem silences warnings originating
# inside the header without weakening any flag for our own code.
target_include_directories(sage_gpu SYSTEM PRIVATE "${Stb_INCLUDE_DIR}")

target_link_libraries(sage_gpu
    PRIVATE
        sage::compile_options
        glfw
        imgui::imgui
        GPUOpen::VulkanMemoryAllocator
        fastgltf::fastgltf
    PUBLIC
        Vulkan::Vulkan
        sage::core
)

# Validation layers and VK_EXT_debug_utils are Debug-only. This is a runtime
# behavior toggle, not a warning/debug-info flag, so it deliberately lives
# here rather than in SageCompileOptions.cmake.
target_compile_definitions(sage_gpu PRIVATE $<$<CONFIG:Debug>:SAGE_VULKAN_VALIDATION>)

# glfw3.h includes <GL/gl.h> unless told otherwise. This is a Vulkan-only
# renderer, so that header is both wrong to pull in and not guaranteed to
# exist -- machines with Mesa dev packages have it, bare CI runners do not.
# Set on the target so a future .cpp including GLFW cannot forget it.
target_compile_definitions(sage_gpu PRIVATE GLFW_INCLUDE_NONE)
