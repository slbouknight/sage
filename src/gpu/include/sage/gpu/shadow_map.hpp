#pragma once

#include <sage/gpu/allocator.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>

namespace sage::gpu {

class Device;

// A depth-only render target holding the scene as the directional light sees
// it, plus the comparison sampler that reads it back.
//
// Fixed size and independent of the swapchain, unlike DepthBuffer, IdBuffer and
// HdrTarget: its resolution is a quality dial for the shadow, not a consequence
// of the window. That also means it survives a resize untouched, so nothing
// here needs recreating on one.
class ShadowMap {
public:
    // D32_SFLOAT for the same reason the main depth buffer uses it -- required
    // by the spec as a depth attachment, and float depth spends its precision
    // where a shadow needs it. No stencil: nothing here tests one.
    static constexpr VkFormat k_format = VK_FORMAT_D32_SFLOAT;

    // 2048 square. Large enough that a hero object does not show stair-stepped
    // shadow edges at a screenshot's resolution, small enough to stay a 16 MiB
    // allocation and one cheap depth pass.
    static constexpr std::uint32_t k_default_resolution = 2048;

    ShadowMap(const Allocator& allocator, const Device& device,
              std::uint32_t resolution = k_default_resolution);
    ~ShadowMap();

    ShadowMap(const ShadowMap&) = delete;
    ShadowMap& operator=(const ShadowMap&) = delete;
    ShadowMap(ShadowMap&&) = delete;
    ShadowMap& operator=(ShadowMap&&) = delete;

    [[nodiscard]] VkImage image() const { return allocation_.image; }
    [[nodiscard]] VkImageView view() const { return view_; }
    [[nodiscard]] VkSampler sampler() const { return sampler_; }
    [[nodiscard]] std::uint32_t resolution() const { return resolution_; }
    [[nodiscard]] VkExtent2D extent() const { return {resolution_, resolution_}; }
    [[nodiscard]] static VkFormat format() { return k_format; }

private:
    const Allocator& allocator_;
    const Device& device_;
    ImageAllocation allocation_;
    VkImageView view_ = VK_NULL_HANDLE;
    // A comparison sampler, not the scene's. Reading a shadow map is not
    // "what depth is stored here" but "is this fragment nearer than what is
    // stored here", and a sampler that answers the second directly gets
    // bilinear filtering of the *comparison result* -- four shadow tests
    // blended in fixed function, which is a free 2x2 PCF per tap.
    VkSampler sampler_ = VK_NULL_HANDLE;
    std::uint32_t resolution_ = 0;
};

}  // namespace sage::gpu
