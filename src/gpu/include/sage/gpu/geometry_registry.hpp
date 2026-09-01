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
// Bump allocation only -- there is no free of an individual mesh. What M6 adds
// is `reset()`, a rewind of the whole buffer, which is all that additive
// loading ever needed: the scene is cleared as a unit, never a mesh at a time.
class GeometryRegistry {
public:
    // Everything needed to draw one mesh.
    struct MeshView {
        VkDeviceAddress vertex_address = 0;
        VkDeviceSize index_offset = 0;
        std::uint32_t index_count = 0;

        // A mesh with no indices cannot be drawn, so a zeroed view doubles as
        // the failure result of add_mesh without a separate sentinel.
        [[nodiscard]] bool valid() const { return index_count > 0; }
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
    //
    // Returns an invalid view when the mesh does not fit. Capacity is a real
    // limit a user can hit by picking a large file, so it is reported rather
    // than asserted -- a viewer that aborts over a file choice is a bug.
    MeshView add_mesh(const void* vertices, VkDeviceSize vertex_bytes, const std::uint32_t* indices,
                      std::uint32_t index_count);

    // Rewinds to empty. Every MeshView handed out before this dangles, so the
    // caller must have dropped them and waited for the device to be idle.
    void reset();

    [[nodiscard]] VkBuffer buffer() const { return allocation_.buffer; }
    [[nodiscard]] VkDeviceSize used() const { return head_; }
    [[nodiscard]] VkDeviceSize capacity() const { return capacity_; }

private:
    const Allocator& allocator_;
    const Uploader& uploader_;
    BufferAllocation allocation_;
    VkDeviceSize capacity_ = 0;
    VkDeviceSize head_ = 0;
    VkDeviceAddress base_address_ = 0;
};

}  // namespace sage::gpu