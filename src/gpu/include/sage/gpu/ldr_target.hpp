#pragma once

#include <sage/gpu/allocator.hpp>

#include <vulkan/vulkan.h>

namespace sage::gpu {

class Device;

// The tonemapped image, sitting between the tonemap and the anti-aliasing pass
// that resolves it to the swapchain.
//
// It exists because FXAA cannot read and write the same image: it samples a
// neighbourhood around every pixel, so writing in place would feed already-
// filtered pixels back into the filter. One of the two ends of that pass has to
// be a separate image, and the swapchain is fixed as the output.
class LdrTarget {
public:
    // UNORM, deliberately, where the swapchain is _SRGB.
    //
    // FXAA measures luma to find edges, and luma has to be perceptual for that
    // to match what the eye calls an edge -- a linear midpoint is nowhere near
    // the visual midpoint. An _SRGB image would defeat that twice over: the
    // hardware would encode on write and decode again on read, handing the
    // filter linear values back. UNORM stores whatever the tonemap wrote,
    // which is sRGB-encoded by hand, and hands it back unchanged.
    static constexpr VkFormat k_format = VK_FORMAT_R8G8B8A8_UNORM;

    LdrTarget(const Allocator& allocator, const Device& device, VkExtent2D extent);
    ~LdrTarget();

    LdrTarget(const LdrTarget&) = delete;
    LdrTarget& operator=(const LdrTarget&) = delete;
    LdrTarget(LdrTarget&&) = delete;
    LdrTarget& operator=(LdrTarget&&) = delete;

    void recreate(VkExtent2D extent);

    [[nodiscard]] VkImage image() const { return allocation_.image; }
    [[nodiscard]] VkImageView view() const { return view_; }
    [[nodiscard]] VkSampler sampler() const { return sampler_; }
    [[nodiscard]] VkExtent2D extent() const { return extent_; }
    [[nodiscard]] static VkFormat format() { return k_format; }

private:
    void create(VkExtent2D extent);
    void destroy();

    const Allocator& allocator_;
    const Device& device_;
    ImageAllocation allocation_;
    VkImageView view_ = VK_NULL_HANDLE;
    // Its own, not the scene's. FXAA's edge walk samples between texel centres
    // and relies on the bilinear blend to read a weighted pair, so filtering
    // must be LINEAR; and the walk runs off the edge of the screen by design,
    // where the scene sampler's REPEAT would wrap round and pull in pixels from
    // the opposite side. Created once and never rebuilt -- it holds no
    // reference to the image, so a resize leaves it valid.
    VkSampler sampler_ = VK_NULL_HANDLE;
    VkExtent2D extent_{};
};

}  // namespace sage::gpu
