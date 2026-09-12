#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/screenshot.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <cstdint>
#include <system_error>
#include <vector>

namespace sage::gpu {

namespace {

constexpr std::size_t k_bytes_per_pixel = 4;
// PNG has no alpha-less RGBA, and every pixel here is opaque, so the encode is
// three channels wide and the fourth is dropped rather than written.
constexpr int k_png_channels = 3;

// Whether the format's first channel is blue. Both of these are 8-bit, 4-channel
// and sRGB-encoded; they differ only in channel order, and PNG wants RGB.
bool is_bgra(VkFormat format) {
    return format == VK_FORMAT_B8G8R8A8_SRGB || format == VK_FORMAT_B8G8R8A8_UNORM;
}

bool is_rgba(VkFormat format) {
    return format == VK_FORMAT_R8G8B8A8_SRGB || format == VK_FORMAT_R8G8B8A8_UNORM;
}

}  // namespace

bool is_capturable_format(VkFormat format) {
    return is_bgra(format) || is_rgba(format);
}

bool write_png(const std::filesystem::path& path, VkExtent2D extent, VkFormat format,
               std::span<const std::byte> pixels) {
    if (!is_capturable_format(format)) {
        SAGE_LOG_ERROR("Cannot capture swapchain format {}: no byte order known for it",
                       static_cast<int>(format));
        return false;
    }

    const std::size_t pixel_count = std::size_t{extent.width} * extent.height;
    if (extent.width == 0 || extent.height == 0 ||
        pixels.size() < pixel_count * k_bytes_per_pixel) {
        SAGE_LOG_ERROR("Screenshot buffer is {} bytes, too small for {}x{}", pixels.size(),
                       extent.width, extent.height);
        return false;
    }

    // Repacked rather than handed to stb as-is: the alpha channel is dropped and
    // BGR is reordered, and doing both in one pass over a buffer sized for the
    // output costs less than the file write that follows it either way.
    std::vector<std::uint8_t> rgb(pixel_count * k_png_channels);
    const std::size_t blue = is_bgra(format) ? 0 : 2;
    const std::size_t red = is_bgra(format) ? 2 : 0;
    for (std::size_t i = 0; i < pixel_count; ++i) {
        const std::byte* source = pixels.data() + (i * k_bytes_per_pixel);
        std::uint8_t* destination = rgb.data() + (i * k_png_channels);
        destination[0] = static_cast<std::uint8_t>(source[red]);
        destination[1] = static_cast<std::uint8_t>(source[1]);
        destination[2] = static_cast<std::uint8_t>(source[blue]);
    }

    std::error_code error;
    if (const std::filesystem::path parent = path.parent_path(); !parent.empty()) {
        std::filesystem::create_directories(parent, error);
        if (error) {
            SAGE_LOG_ERROR("Could not create {}: {}", parent.string(), error.message());
            return false;
        }
    }

    const auto width = static_cast<int>(extent.width);
    const auto height = static_cast<int>(extent.height);
    const int stride = width * k_png_channels;
    if (stbi_write_png(path.c_str(), width, height, k_png_channels, rgb.data(), stride) == 0) {
        SAGE_LOG_ERROR("Could not write {}", path.string());
        return false;
    }

    SAGE_LOG_INFO("Wrote {} ({}x{})", path.string(), extent.width, extent.height);
    return true;
}

}  // namespace sage::gpu
