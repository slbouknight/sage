#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/depth_buffer.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/vk_check.hpp>

#include <array>

namespace sage::gpu {

namespace {

const char* depth_format_name(VkFormat format) {
    switch (format) {
        case VK_FORMAT_D32_SFLOAT:
            return "D32_SFLOAT";
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return "D32_SFLOAT_S8_UINT";
        case VK_FORMAT_D24_UNORM_S8_UINT:
            return "D24_UNORM_S8_UINT";
        default:
            return "unknown";
    }
}

// The spec only guarantees that one of D32_SFLOAT / X8_D24_UNORM_PACK32 works
// as a depth attachment, so query instead of assuming. Highest precision and
// stencil-free first -- nothing here needs stencil.
VkFormat select_depth_format(VkPhysicalDevice physical_device) {
    constexpr std::array<VkFormat, 3> k_candidates{
        VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};

    for (const VkFormat format : k_candidates) {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(physical_device, format, &props);
        if ((props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0U) {
            return format;
        }
    }

    SAGE_VERIFY(false, "No supported depth attachment format");
    return VK_FORMAT_UNDEFINED;
}

}  // namespace

DepthBuffer::DepthBuffer(const Allocator& allocator, const Device& device, VkExtent2D extent)
    : allocator_(allocator), device_(device) {
    format_ = select_depth_format(device_.physical_device());
    SAGE_LOG_INFO("Depth format: {}", depth_format_name(format_));
    create(extent);
}

void DepthBuffer::create(VkExtent2D extent) {
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = format_;
    image_info.extent = {extent.width, extent.height, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    allocation_ = allocator_.create_device_local_image(image_info);

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = allocation_.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format_;
    // Depth aspect only. Even if a stencil-carrying format was selected,
    // nothing here reads or writes stencil.
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device_.handle(), &view_info, nullptr, &view_));
}

void DepthBuffer::destroy() {
    if (view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_.handle(), view_, nullptr);
        view_ = VK_NULL_HANDLE;
    }
    allocator_.destroy_image(allocation_);
    allocation_ = {};
}

void DepthBuffer::recreate(VkExtent2D extent) {
    device_.wait_idle();
    destroy();
    create(extent);
}

DepthBuffer::~DepthBuffer() {
    destroy();
}

}  // namespace sage::gpu