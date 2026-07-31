#pragma once

#include <sage/core/assert.hpp>

#include <vulkan/vulkan.h>

namespace sage::gpu {

const char* vk_result_to_string(VkResult result);

void vk_check_failed(VkResult result, const char* expr, const char* file, int line);

}  // namespace sage::gpu

// Wraps a Vulkan call that must succeed. Non-VK_SUCCESS results abort with the
// result name and call site. Results that are expected to be handled by the
// caller (VK_SUBOPTIMAL_KHR, VK_ERROR_OUT_OF_DATE_KHR from acquire/present)
// must not go through this macro.
#define VK_CHECK(expr)                                                                \
    do {                                                                              \
        const VkResult sage_vk_result_ = (expr);                                      \
        if (sage_vk_result_ != VK_SUCCESS) {                                          \
            ::sage::gpu::vk_check_failed(sage_vk_result_, #expr, __FILE__, __LINE__); \
        }                                                                             \
    } while (0)
