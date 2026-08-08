#pragma once

#include <sage/gpu/allocator.hpp>

#include <vulkan/vulkan.h>

namespace sage::gpu {

class Device;

// Swapchain-sized depth attachment. Recreated alongside the swapchain, and
// shaped like Swapchain so Application handles both identically on resize.
class DepthBuffer {
public:
    DepthBuffer(const Allocator& allocator, const Device& device, VkExtent2D extent);
    ~DepthBuffer();

    DepthBuffer(const DepthBuffer&) = delete;
    DepthBuffer& operator=(const DepthBuffer&) = delete;
    DepthBuffer(DepthBuffer&&) = delete;
    DepthBuffer& operator=(DepthBuffer&&) = delete;

    void recreate(VkExtent2D extent);
    [[nodiscard]] VkImage image() const { return allocation_.image; }
    [[nodiscard]] VkImageView view() const { return view_; }

    // Fixed for the lifetime of the device: the pipeline bakes it in, so it
    // must not change across a recreate.
    [[nodiscard]] VkFormat format() const { return format_; }

private:
    void create(VkExtent2D extent);
    void destroy();

    const Allocator& allocator_;
    const Device& device_;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    ImageAllocation allocation_;
    VkImageView view_ = VK_NULL_HANDLE;
};

} // namespace sage::gpu