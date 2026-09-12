#include <sage/core/assert.hpp>
#include <sage/gpu/buffer.hpp>
#include <sage/gpu/device.hpp>

#include <cstddef>
#include <cstring>

namespace sage::gpu {
Buffer::Buffer(const Allocator& allocator, const Device& device, VkDeviceSize size,
               VkBufferUsageFlags usage, BufferAccess access)
    : allocator_(allocator), size_(size) {
    allocation_ = access == BufferAccess::host_read ? allocator_.create_readback_buffer(size, usage)
                                                    : allocator_.create_mapped_buffer(size, usage);

    if ((usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0U) {
        VkBufferDeviceAddressInfo address_info{};
        address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        address_info.buffer = allocation_.buffer;
        address_ = vkGetBufferDeviceAddress(device.handle(), &address_info);
    }
}

void Buffer::write(const void* data, VkDeviceSize size, VkDeviceSize offset) const {
    SAGE_VERIFY(offset + size <= size_, "Buffer write exceeds allocation size");
    SAGE_VERIFY(allocation_.mapped != nullptr, "Buffer is not host-visible");

    std::memcpy(static_cast<std::byte*>(allocation_.mapped) + offset, data, size);
    allocator_.flush(allocation_, size, offset);
}

void Buffer::read(void* destination, VkDeviceSize size, VkDeviceSize offset) const {
    SAGE_VERIFY(offset + size <= size_, "Buffer read exceeds allocation size");
    SAGE_VERIFY(allocation_.mapped != nullptr, "Buffer is not host-visible");

    // Before the memcpy, not after: on non-coherent memory the device's writes
    // are not guaranteed visible to the host until the range is invalidated.
    allocator_.invalidate(allocation_, size, offset);
    std::memcpy(destination, static_cast<const std::byte*>(allocation_.mapped) + offset, size);
}

Buffer::~Buffer() {
    allocator_.destroy_buffer(allocation_);
}

}  // namespace sage::gpu