#pragma once

#include <sage/gpu/allocator.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <filesystem>

namespace sage::gpu {

class Device;
class Uploader;

// A sampled 2D image with a full mip chain, decoded from a file on disk.
// Owns the image, its allocation and its view. The sampler is deliberately not here:
// one sampler is shared by every texture (see Sampler).
class Texture {
public:
    Texture(const Allocator& allocator, const Device& device, const Uploader& uploader,
            const std::filesystem::path& path);
    ~Texture();

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;
    Texture(Texture&&) = delete;
    Texture& operator=(Texture&&) = delete;

    [[nodiscard]] VkImageView view() const { return view_; }
    [[nodiscard]] std::uint32_t mip_levels() const { return mip_levels_; }

private:
    const Allocator& allocator_;
    const Device& device_;
    ImageAllocation allocation_;
    VkImageView view_ = VK_NULL_HANDLE;
    std::uint32_t mip_levels_ = 1;
};

}  // namespace sage::gpu