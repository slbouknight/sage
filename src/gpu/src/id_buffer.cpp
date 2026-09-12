#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/id_buffer.hpp>
#include <sage/gpu/vk_check.hpp>

namespace sage::gpu {

IdBuffer::IdBuffer(const Allocator& allocator, const Device& device, VkExtent2D extent)
    : allocator_(allocator), device_(device) {
    // Checked once rather than trusted: the required-format table makes this
    // safe in principle, and a driver that disagrees should say so here rather
    // than at the first vkCmdBeginRendering.
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(device_.physical_device(), k_format, &props);
    SAGE_VERIFY((props.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) != 0U,
                "R32_UINT is not usable as a colour attachment");
    SAGE_VERIFY((props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0U,
                "R32_UINT is not sampleable");

    create(extent);
    SAGE_LOG_INFO("Object-id buffer: R32_UINT, {}x{}", extent.width, extent.height);
}

void IdBuffer::create(VkExtent2D extent) {
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
    // TRANSFER_SRC for the one-texel picking readback; SAMPLED for the outline
    // pass that reads this same image back as a texture.
    image_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                       VK_IMAGE_USAGE_SAMPLED_BIT;
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

void IdBuffer::destroy() {
    if (view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_.handle(), view_, nullptr);
        view_ = VK_NULL_HANDLE;
    }
    allocator_.destroy_image(allocation_);
    allocation_ = {};
}

void IdBuffer::recreate(VkExtent2D extent) {
    device_.wait_idle();
    destroy();
    create(extent);
}

IdBuffer::~IdBuffer() {
    destroy();
}

}  // namespace sage::gpu
