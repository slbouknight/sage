#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/hdr_target.hpp>
#include <sage/gpu/vk_check.hpp>

namespace sage::gpu {

HdrTarget::HdrTarget(const Allocator& allocator, const Device& device, VkExtent2D extent)
    : allocator_(allocator), device_(device) {
    // Same reasoning as IdBuffer: the required-format table makes this safe in
    // principle, and a driver that disagrees should say so here rather than at
    // the first vkCmdBeginRendering.
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(device_.physical_device(), k_format, &props);
    SAGE_VERIFY((props.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) != 0U,
                "R16G16B16A16_SFLOAT is not usable as a colour attachment");
    SAGE_VERIFY((props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0U,
                "R16G16B16A16_SFLOAT is not sampleable");

    create(extent);
    SAGE_LOG_INFO("HDR target: R16G16B16A16_SFLOAT, {}x{}", extent.width, extent.height);
}

void HdrTarget::create(VkExtent2D extent) {
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
    // SAMPLED is what the tonemap pass reads it back through. No TRANSFER_SRC:
    // a capture wants the resolved image, not the unbounded values behind it,
    // and there is no 8-bit file format that could hold these anyway.
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

void HdrTarget::destroy() {
    if (view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_.handle(), view_, nullptr);
        view_ = VK_NULL_HANDLE;
    }
    allocator_.destroy_image(allocation_);
    allocation_ = {};
}

void HdrTarget::recreate(VkExtent2D extent) {
    device_.wait_idle();
    destroy();
    create(extent);
}

HdrTarget::~HdrTarget() {
    destroy();
}

}  // namespace sage::gpu
