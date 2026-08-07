#pragma once

#include <vulkan/vulkan.h>

// Opaque VmaAllocator handle, declared the same way Vulkan declares its own
// dispatchable handles. Keeps vk_mem_alloc.h confined to allocator.cpp.
struct VmaAllocator_T;
struct VmaAllocation_T;

// A buffer plus the allocation backing it. `mapped` is a persistently mapped
// host pointer, null for device local allocations.
struct BufferAllocation {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation_T* allocation = nullptr;
    void* mapped = nullptr;
};

namespace sage::gpu {

class Device;
class Instance;

// M1 scope is lifecycle only. Buffer/image suballocation arrives with M3's
// geometry registry.
class Allocator {
public:
    Allocator(const Instance& instance, const Device& device);
    ~Allocator();

    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;
    Allocator(Allocator&&) = delete;
    Allocator& operator=(Allocator&&) = delete;

    [[nodiscard]] VmaAllocator_T* handle() const { return allocator_; }

    // Host-visible and persistently mapped. M3 will replace this with device local
    // allocations fed by transfer-queue staging uploads.
    [[nodiscard]] BufferAllocation create_mapped_buffer(VkDeviceSize size,
                                                        VkBufferUsageFlags usage) const;
    void flush(const BufferAllocation& allocation, VkDeviceSize size) const;
    void destroy_buffer(const BufferAllocation& allocation) const;

private:
    VmaAllocator_T* allocator_ = nullptr;
};

}  // namespace sage::gpu
