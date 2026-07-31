#pragma once

#include <vulkan/vulkan.h>

// Opaque VmaAllocator handle, declared the same way Vulkan declares its own
// dispatchable handles. Keeps vk_mem_alloc.h confined to allocator.cpp.
struct VmaAllocator_T;

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

private:
    VmaAllocator_T* allocator_ = nullptr;
};

}  // namespace sage::gpu
