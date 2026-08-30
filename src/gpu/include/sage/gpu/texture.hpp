#pragma once

#include <sage/gpu/allocator.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <filesystem>

namespace sage::gpu {

class Device;
class Uploader;

// Base color and emissive hold color, and must decode sRGB -> linear on read.
// Normal and metallic-roughness hold measurements: a decode curve applied to a
// packed normal or a roughness corrupts it. The result looks almost right,
// which is what makes this worth naming at the call site.
enum class TextureColorSpace {
    srgb,
    linear,
};

// A sampled 2D image with a full mip chain, decoded from a file on disk.
// Owns the image, its allocation and its view. The sampler is deliberately not here:
// one sampler is shared by every texture (see Sampler).
class Texture {
public:
    // Decodes an image file from disk.
    Texture(const Allocator& allocator, const Device& device, const Uploader& uploader,
            const std::filesystem::path& path, TextureColorSpace color_space);

    // Wraps pixels already in memory: RGBA8, tightly packed, extent texels.
    // Used for the 1x1 fallback, which has no file to decode.
    Texture(const Allocator& allocator, const Device& device, const Uploader& uploader,
            const std::uint8_t* pixels, VkExtent2D extent, TextureColorSpace color_space);

    ~Texture();

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;
    Texture(Texture&&) = delete;
    Texture& operator=(Texture&&) = delete;

    [[nodiscard]] VkImageView view() const { return view_; }
    [[nodiscard]] std::uint32_t mip_levels() const { return mip_levels_; }

private:
    // Everything both constructors share, once the pixels exist either way.
    void create(const Uploader& uploader, const std::uint8_t* pixels, VkExtent2D extent,
                TextureColorSpace color_space);
    const Allocator& allocator_;
    const Device& device_;
    ImageAllocation allocation_;
    VkImageView view_ = VK_NULL_HANDLE;
    std::uint32_t mip_levels_ = 1;
};

}  // namespace sage::gpu