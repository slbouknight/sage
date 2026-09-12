#pragma once

#include <sage/gpu/allocator.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>

namespace sage::gpu {

class Device;

// Swapchain-sized R32_UINT attachment holding one object id per pixel, written
// alongside colour by the same draw. Shaped like DepthBuffer so Application
// recreates all three the same way on resize.
//
// One buffer serves two features. Copying a single texel back gives cursor
// picking; sampling it and comparing neighbours gives an outline wherever the
// id changes. The alternative for the outline would be stencil, which the
// D32_SFLOAT depth format does not carry.
class IdBuffer {
public:
    // Guaranteed as a colour attachment by the spec's required-format table,
    // and verified on NVIDIA, RADV and llvmpipe. Integer, so no filtering and
    // no blending -- an id must survive as the exact value written.
    static constexpr VkFormat k_format = VK_FORMAT_R32_UINT;

    // Written when nothing was drawn to a pixel, so it is the "no object"
    // answer a pick gets for empty space. Ids are therefore node index + 1.
    static constexpr std::uint32_t k_null_id = 0;

    IdBuffer(const Allocator& allocator, const Device& device, VkExtent2D extent);
    ~IdBuffer();

    IdBuffer(const IdBuffer&) = delete;
    IdBuffer& operator=(const IdBuffer&) = delete;
    IdBuffer(IdBuffer&&) = delete;
    IdBuffer& operator=(IdBuffer&&) = delete;

    void recreate(VkExtent2D extent);

    [[nodiscard]] VkImage image() const { return allocation_.image; }
    [[nodiscard]] VkImageView view() const { return view_; }
    [[nodiscard]] VkExtent2D extent() const { return extent_; }
    [[nodiscard]] static VkFormat format() { return k_format; }

private:
    void create(VkExtent2D extent);
    void destroy();

    const Allocator& allocator_;
    const Device& device_;
    ImageAllocation allocation_;
    VkImageView view_ = VK_NULL_HANDLE;
    VkExtent2D extent_{};
};

}  // namespace sage::gpu
