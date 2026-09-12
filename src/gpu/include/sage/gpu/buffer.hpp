#pragma once

#include <sage/gpu/allocator.hpp>

#include <vulkan/vulkan.h>

namespace sage::gpu {
class Device;

// Which way the host touches the buffer. This is not a hint: it decides the
// memory type VMA selects, and reading from a write-optimised allocation is
// correct but slow enough to be a bug.
enum class BufferAccess {
    host_write,
    host_read,
};

// A single VMA-backed buffer. Deliberately minimal: no suballocation, no staging, no growth.
// Those belong in M3's future geometry registry.
class Buffer {
public:
    Buffer(const Allocator& allocator, const Device& device, VkDeviceSize size,
           VkBufferUsageFlags usage, BufferAccess access = BufferAccess::host_write);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) = delete;
    Buffer& operator=(Buffer&&) = delete;

    // 'offset' selects a sub-region; the default overwrites from the start.
    void write(const void* data, VkDeviceSize size, VkDeviceSize offset = 0) const;

    // Copies out of the mapping, invalidating first so device writes are
    // visible. The caller is responsible for having waited on whatever
    // submission produced the contents; this does no synchronisation.
    void read(void* destination, VkDeviceSize size, VkDeviceSize offset = 0) const;

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