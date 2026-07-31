#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace sage::gpu {

struct QueueFamilyIndices {
    std::optional<std::uint32_t> graphics;
    std::optional<std::uint32_t> present;
    // Falls back to the graphics family when no transfer-only family exists.
    std::optional<std::uint32_t> transfer;

    [[nodiscard]] bool complete() const {
        return graphics.has_value() && present.has_value() && transfer.has_value();
    }
};

struct PhysicalDeviceInfo {
    VkPhysicalDevice handle = VK_NULL_HANDLE;
    QueueFamilyIndices queues;
    VkPhysicalDeviceProperties properties{};
};

// Pure helpers, split out from the enumeration path so they can be unit tested
// against fabricated data without a live VkInstance.

// present_support[i] is whether family i can present to the target surface.
QueueFamilyIndices select_queue_families(const std::vector<VkQueueFamilyProperties>& families,
                                         const std::vector<bool>& present_support);

std::uint64_t device_local_memory_bytes(const VkPhysicalDeviceMemoryProperties& memory);

std::uint64_t score_device(const VkPhysicalDeviceProperties& properties,
                           const VkPhysicalDeviceMemoryProperties& memory);

PhysicalDeviceInfo select_physical_device(VkInstance instance, VkSurfaceKHR surface);

}  // namespace sage::gpu
