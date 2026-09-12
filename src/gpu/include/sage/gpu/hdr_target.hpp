#pragma once

#include <sage/gpu/allocator.hpp>

#include <vulkan/vulkan.h>

namespace sage::gpu {

class Device;

// The colour target the scene is shaded into, ahead of the tonemap that resolves
// it to the swapchain. Shaped like DepthBuffer and IdBuffer so Application
// recreates all four the same way on resize.
//
// Shading has always produced values above 1.0 -- a specular highlight under a
// light of intensity 2 trivially does -- and writing them straight to an
// 8-bit-per-channel swapchain clamped them at the attachment. Two highlights of
// very different energy came out the same flat white. A half-float target holds
// them instead, and the tonemap decides how they land in [0, 1] once, with the
// whole image in hand.
class HdrTarget {
public:
    // Half float, not full: the spec's required-format table guarantees this one
    // as a colour attachment and as a sampled image, it holds far more range
    // than the 8 bits it replaces, and it costs half the bandwidth of R32G32B32A32
    // for precision no display can show. Verified on NVIDIA, RADV and llvmpipe.
    static constexpr VkFormat k_format = VK_FORMAT_R16G16B16A16_SFLOAT;

    HdrTarget(const Allocator& allocator, const Device& device, VkExtent2D extent);
    ~HdrTarget();

    HdrTarget(const HdrTarget&) = delete;
    HdrTarget& operator=(const HdrTarget&) = delete;
    HdrTarget(HdrTarget&&) = delete;
    HdrTarget& operator=(HdrTarget&&) = delete;

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
