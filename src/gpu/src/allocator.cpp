#include <sage/core/log.hpp>
#include <sage/gpu/allocator.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/instance.hpp>
#include <sage/gpu/vk_check.hpp>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

namespace sage::gpu {

Allocator::Allocator(const Instance& instance, const Device& device) {
    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo create_info{};
    create_info.vulkanApiVersion = VK_API_VERSION_1_3;
    create_info.instance = instance.handle();
    create_info.physicalDevice = device.physical_device();
    create_info.device = device.handle();
    create_info.pVulkanFunctions = &functions;
    // Must match the device feature; without it VMA will not add
    // VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT to allocations M2 needs.
    create_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    VK_CHECK(vmaCreateAllocator(&create_info, &allocator_));
    SAGE_LOG_INFO("VMA allocator created");
}

Allocator::~Allocator() {
    if (allocator_ != nullptr) {
        vmaDestroyAllocator(allocator_);
    }
}

}  // namespace sage::gpu
