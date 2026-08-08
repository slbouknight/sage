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

BufferAllocation Allocator::create_mapped_buffer(VkDeviceSize size,
                                                 VkBufferUsageFlags usage) const {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_info.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    BufferAllocation result;
    VmaAllocationInfo info{};
    VK_CHECK(vmaCreateBuffer(allocator_, &buffer_info, &alloc_info, &result.buffer,
                             &result.allocation, &info));
    result.mapped = info.pMappedData;
    return result;
}

BufferAllocation Allocator::create_device_local_buffer(VkDeviceSize size, VkBufferUsageFlags usage) const
{
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;

    BufferAllocation result;
    VK_CHECK(vmaCreateBuffer(allocator_, &buffer_info, &alloc_info, &result.buffer, &result.allocation, nullptr));
    return result;
}

void Allocator::flush(const BufferAllocation& allocation, VkDeviceSize size) const {
    VK_CHECK(vmaFlushAllocation(allocator_, allocation.allocation, 0, size));
}

void Allocator::destroy_buffer(const BufferAllocation& allocation) const {
    if (allocation.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, allocation.buffer, allocation.allocation);
    }
}

ImageAllocation Allocator::create_device_local_image(const VkImageCreateInfo& info) const {
    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    // A full-screen render target is large and long-lived, so a dedicated
    // allocation is preferable to carving it out of a shared block.
    alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    ImageAllocation result;
    VK_CHECK(vmaCreateImage(allocator_, &info, &alloc_info, &result.image, &result.allocation, nullptr));
    return result;
}

void Allocator::destroy_image(const ImageAllocation& allocation) const {
    if (allocation.image != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, allocation.image, allocation.allocation);
    }
}

}  // namespace sage::gpu
