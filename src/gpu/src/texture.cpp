// clang-format off
// STB_IMAGE_IMPLEMENTATION must be defined in exactly one translation unit,
// immediately before the header it configures. Include sorting is disabled
// here so the pair cannot be separated.
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
// clang-format on

#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/texture.hpp>
#include <sage/gpu/uploader.hpp>
#include <sage/gpu/vk_check.hpp>

#include <algorithm>
#include <bit>
#include <cstdint>

namespace sage::gpu {

namespace {

// sRGB, so the sampler converts to linear on read and shading happens in
// linear space. A UNORM format here would light sRGB-encoded values directly,
// which is subtly wrong everywhere and obviously wrong in shadowed areas.
constexpr VkFormat k_format = VK_FORMAT_R8G8B8A8_SRGB;

}  // namespace

Texture::Texture(const Allocator& allocator, const Device& device, const Uploader& uploader,
                 const std::filesystem::path& path)
    : allocator_(allocator), device_(device) {
    // Generating mips by blit needs all three of these, and they are per-format
    // and per-driver guarantees rather than universal ones.
    VkFormatProperties format_properties{};
    vkGetPhysicalDeviceFormatProperties(device_.physical_device(), k_format, &format_properties);
    constexpr VkFormatFeatureFlags k_required = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                                                VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                                                VK_FORMAT_FEATURE_BLIT_DST_BIT;
    SAGE_VERIFY((format_properties.optimalTilingFeatures & k_required) == k_required,
                "Texture format cannot be linearly blitted; mip generation would be invalid");

    int width = 0;
    int height = 0;
    int channels_in_file = 0;
    // STBI_rgb_alpha forces four channels whatever the file holds. This is not
    // convenience: R8G8B8_SRGB reports no optimal-tiling features at all on
    // this hardware, so a three-channel image is simply not sampleable.
    stbi_uc* pixels = stbi_load(path.c_str(), &width, &height, &channels_in_file, STBI_rgb_alpha);
    if (pixels == nullptr) {
        SAGE_LOG_ERROR("Could not decode image {}: {}", path.string(), stbi_failure_reason());
    }
    SAGE_VERIFY(pixels != nullptr, "Failed to decode texture");
    SAGE_VERIFY(width > 0 && height > 0, "Texture has zero extent");

    const auto extent_width = static_cast<std::uint32_t>(width);
    const auto extent_height = static_cast<std::uint32_t>(height);

    // bit_width(x) is floor(log2(x)) + 1, which is exactly the number of times
    // the larger axis can halve before reaching 1x1.
    mip_levels_ = static_cast<std::uint32_t>(std::bit_width(std::max(extent_width, extent_height)));

    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = k_format;
    image_info.extent = {extent_width, extent_height, 1};
    image_info.mipLevels = mip_levels_;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    // TRANSFER_SRC as well as DST: building the mip chain blits out of level
    // i-1 into level i, so the image is its own blit source.
    image_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                       VK_IMAGE_USAGE_SAMPLED_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    allocation_ = allocator_.create_device_local_image(image_info);

    // Four bytes per texel, forced above. Widened before multiplying so a large
    // texture cannot overflow an int on the way to a VkDeviceSize.
    const VkDeviceSize size =
        static_cast<VkDeviceSize>(extent_width) * static_cast<VkDeviceSize>(extent_height) * 4;
    uploader.upload_to_image(allocation_.image, {extent_width, extent_height}, mip_levels_, pixels,
                             size);

    // Safe immediately: upload_to_image blocks until the copy has completed.
    stbi_image_free(pixels);

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = allocation_.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = k_format;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    // Every level, or the sampler cannot select between them and maxLod is moot.
    view_info.subresourceRange.levelCount = mip_levels_;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device_.handle(), &view_info, nullptr, &view_));

    SAGE_LOG_INFO("Texture {}: {}x{}, {} mip levels, {} channels in file, loaded as RGBA",
                  path.filename().string(), extent_width, extent_height, mip_levels_,
                  channels_in_file);
}

Texture::~Texture() {
    if (view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_.handle(), view_, nullptr);
    }
    allocator_.destroy_image(allocation_);
}

}  // namespace sage::gpu