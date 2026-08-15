#pragma once

#include <sage/gpu/allocator.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>

namespace sage::gpu {

class Device;
class Uploader;

// One device-local buffer holding every mesh, bump-suballocated. Vertices are
// read by buffer device address; indices are bound with vkCmdBindIndexBuffer,
// which is fixed-function and takes a buffer plus offset rather than a pointer.
//
// Bump allocation only -- there is no free. M3's geometry is static, and a
// free list would be machinery without a user.
class GeometryRegistry {
public:
    // Everything needed to draw one mesh.
    struct MeshView {
        VkDeviceAddress vertex_address = 0;
        VkDeviceSize index_offset = 0;
        std::uint32_t index_count = 0;
    };

    GeometryRegistry(const Allocator& allocator, const Device& device, const Uploader& uploader,
                     VkDeviceSize capacity);
    ~GeometryRegistry();

    GeometryRegistry(const GeometryRegistry&) = delete;
    GeometryRegistry& operator=(const GeometryRegistry&) = delete;
    GeometryRegistry(GeometryRegistry&&) = delete;
    GeometryRegistry& operator=(GeometryRegistry&&) = delete;

    // Blocks until the mesh is resident and owned by the graphics family.
    // Vertex layout is the caller's concern; the registry only moves bytes.
    MeshView add_mesh(const void* vertices, VkDeviceSize vertex_bytes, const std::uint32_t* indices,
                      std::uint32_t index_count);

    [[nodiscard]] VkBuffer buffer() const { return allocation_.buffer; }

private:
    VkDeviceSize allocate(VkDeviceSize size, VkDeviceSize alignment);

    const Allocator& allocator_;
    const Uploader& uploader_;
    BufferAllocation allocation_;
    VkDeviceSize capacity_ = 0;
    VkDeviceSize head_ = 0;
    VkDeviceAddress base_address_ = 0;
};

}  // namespace sage::gpu