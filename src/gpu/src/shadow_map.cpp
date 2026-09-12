#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/shadow_map.hpp>
#include <sage/gpu/vk_check.hpp>

namespace sage::gpu {

ShadowMap::ShadowMap(const Allocator& allocator, const Device& device, std::uint32_t resolution)
    : allocator_(allocator), device_(device), resolution_(resolution) {
    SAGE_VERIFY(resolution_ > 0, "ShadowMap: zero resolution");

    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(device_.physical_device(), k_format, &props);
    SAGE_VERIFY(
        (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0U,
        "D32_SFLOAT is not usable as a depth attachment");
    SAGE_VERIFY((props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0U,
                "D32_SFLOAT is not sampleable");

    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = k_format;
    image_info.extent = {resolution_, resolution_, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    allocation_ = allocator_.create_device_local_image(image_info);

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = allocation_.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = k_format;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device_.handle(), &view_info, nullptr, &view_));

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    // LINEAR with compareEnable filters the boolean results of four depth
    // tests, not the depths themselves -- averaging raw depths would compare
    // against a surface that exists nowhere in the scene.
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    // CLAMP_TO_BORDER with an opaque-white border: a fragment outside the
    // light's frustum samples the border and comes back fully lit. CLAMP_TO_EDGE
    // would instead smear the edge texel's occlusion outwards, streaking
    // shadows across everything beyond the map.
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    sampler_info.anisotropyEnable = VK_FALSE;
    sampler_info.compareEnable = VK_TRUE;
    // The reference depth is the fragment's own. LESS_OR_EQUAL returns 1 when
    // the fragment is at or in front of the recorded depth, i.e. lit.
    sampler_info.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    sampler_info.minLod = 0.0F;
    sampler_info.maxLod = 0.0F;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    VK_CHECK(vkCreateSampler(device_.handle(), &sampler_info, nullptr, &sampler_));

    const VkDeviceSize bytes = VkDeviceSize{resolution_} * resolution_ * sizeof(float);
    SAGE_LOG_INFO("Shadow map: D32_SFLOAT {}x{} ({} MiB), comparison sampler", resolution_,
                  resolution_, bytes / (1024 * 1024));
}

ShadowMap::~ShadowMap() {
    if (sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_.handle(), sampler_, nullptr);
    }
    if (view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_.handle(), view_, nullptr);
    }
    allocator_.destroy_image(allocation_);
}

}  // namespace sage::gpu
