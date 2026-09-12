#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <filesystem>
#include <span>

namespace sage::gpu {

// Encodes a block of swapchain pixels, already copied back to host memory, as a
// PNG.
//
// The bytes come out of the swapchain unconverted, and the swapchain is an
// _SRGB format -- so they are already sRGB-encoded 8-bit values, which is
// exactly what a PNG holds. No transfer function is applied here; applying one
// would be applying it twice.
//
// Rows are assumed tightly packed at `extent.width` pixels, which is what
// vkCmdCopyImageToBuffer produces with bufferRowLength left at 0.
//
// Returns false and logs on a write failure -- a full disk or an unwritable
// directory is a message, not a reason to bring the editor down.
[[nodiscard]] bool write_png(const std::filesystem::path& path, VkExtent2D extent, VkFormat format,
                             std::span<const std::byte> pixels);

// Whether write_png knows how to read a given swapchain format. Checked before
// a capture is recorded, so an unsupported surface format costs a message
// rather than a copy that is then thrown away.
[[nodiscard]] bool is_capturable_format(VkFormat format);

}  // namespace sage::gpu
