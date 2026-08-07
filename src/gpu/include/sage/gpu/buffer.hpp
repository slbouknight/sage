#pragma once

#include <sage/gpu/allocator.hpp>

#include <vulkan/vulkan.h>

namespace sage::gpu {
class Device;

// A single VMA-backed buffer. Deliberately minimal: no suballocation, no staging, no growth.
// Those belong in M3's future geometry registry.
class Buffer {
public:
    Buffer(const Allocator& allocator, const Device& device, VkDeviceSize size,
           VkBufferUsageFlags usage);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) = delete;
    Buffer& operator=(Buffer&&) = delete;

    void write(const void* data, VkDeviceSize size) const;

    [[nodiscard]] VkBuffer handle() const { return allocation_.buffer; }
    [[nodiscard]] VkDeviceSize size() const { return size_; }

    // Zero unless created with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
    [[nodiscard]] VkDeviceAddress device_address() const { return address_; }

private:
    const Allocator& allocator_;
    BufferAllocation allocation_;
    VkDeviceSize size_ = 0;
    VkDeviceAddress address_ = 0;
};

}  // namespace sage::gpu