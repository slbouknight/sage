#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/ldr_target.hpp>
#include <sage/gpu/vk_check.hpp>

namespace sage::gpu {

LdrTarget::LdrTarget(const Allocator& allocator, const Device& device, VkExtent2D extent)
    : allocator_(allocator), device_(device) {
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(device_.physical_device(), k_format, &props);
    SAGE_VERIFY((props.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) != 0U,
                "R8G8B8A8_UNORM is not usable as a colour attachment");
    // Not merely sampleable: FXAA needs the bilinear blend specifically, and
    // that is a separate format feature a driver could in principle withhold.
    SAGE_VERIFY(
        (props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0U,
        "R8G8B8A8_UNORM does not support linear filtering");

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.anisotropyEnable = VK_FALSE;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.minLod = 0.0F;
    sampler_info.maxLod = 0.0F;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    VK_CHECK(vkCreateSampler(device_.handle(), &sampler_info, nullptr, &sampler_));

    create(extent);
    SAGE_LOG_INFO("LDR target: R8G8B8A8_UNORM, {}x{}, linear clamped sampler", extent.width,
                  extent.height);
}

void LdrTarget::create(VkExtent2D extent) {
    extent_ = extent;

    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = k_format;
    image_info.extent = {extent.width, extent.height, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    allocation_ = allocator_.create_device_local_image(image_info);

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = allocation_.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = k_format;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device_.handle(), &view_info, nullptr, &view_));
}

void LdrTarget::destroy() {
    if (view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_.handle(), view_, nullptr);
        view_ = VK_NULL_HANDLE;
    }
    allocator_.destroy_image(allocation_);
    allocation_ = {};
}

void LdrTarget::recreate(VkExtent2D extent) {
    device_.wait_idle();
    destroy();
    create(extent);
}

LdrTarget::~LdrTarget() {
    destroy();
    if (sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_.handle(), sampler_, nullptr);
    }
}

}  // namespace sage::gpu
