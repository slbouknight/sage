find_package(Vulkan QUIET)
if(NOT Vulkan_FOUND)
    if(DEFINED ENV{VULKAN_SDK})
        message(FATAL_ERROR "Vulkan not found under VULKAN_SDK=$ENV{VULKAN_SDK}")
    endif()
    message(FATAL_ERROR
        "Vulkan not found. Source the LunarG SDK's setup-env.sh before configuring, e.g.\n"
        "    source ~/vulkansdk/<version>/setup-env.sh")
endif()
find_package(glfw3 CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

add_library(sage_gpu STATIC
    src/allocator.cpp
    src/device.cpp
    src/frame_pacer.cpp
    src/instance.cpp
    src/physical_device.cpp
    src/surface.cpp
    src/swapchain.cpp
    src/vk_check.cpp
    src/window.cpp
)
add_library(sage::gpu ALIAS sage_gpu)

target_include_directories(sage_gpu PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(sage_gpu
    PRIVATE
        sage::compile_options
        sage::core
        glfw
        GPUOpen::VulkanMemoryAllocator
    PUBLIC
        Vulkan::Vulkan
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
