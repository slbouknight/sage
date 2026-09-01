#pragma once

#include <sage/gpu/allocator.hpp>
#include <sage/gpu/material.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace sage::gpu {

class BindlessSet;
class Uploader;

// The scene's material table: one device-local storage buffer, registered once
// into the bindless set and indexed from the shader by a per-draw index.
//
// Device-local rather than host-visible because the fragment shader reads it once
// per fragment. A host-visible buffer would pull those reads across the PCIe. It
// is written only at load time, so the upload cost is irrelevant.
//
// Bump-allocated with a rewind, matching GeometryRegistry and TextureRegistry:
// each load appends and is told where its block starts.
class MaterialRegistry {
public:
    // Slot in BindlessSet's storage-buffer array. Fixed rather than allocated:
    // there is one material table, and the shader needs to name it.
    static constexpr std::uint32_t k_storage_slot = 0;

    MaterialRegistry(const Allocator& allocator, const Uploader& uploader,
                     const BindlessSet& bindless_set, std::uint32_t capacity);
    ~MaterialRegistry();

    MaterialRegistry(const MaterialRegistry&) = delete;
    MaterialRegistry& operator=(const MaterialRegistry&) = delete;
    MaterialRegistry(MaterialRegistry&&) = delete;
    MaterialRegistry& operator=(MaterialRegistry&&) = delete;

    // Appends a block and returns the index its first material landed at. A
    // loader's local material index plus that base is the table index a draw
    // puts in its push constants.
    //
    // Only the new block is uploaded, at its own byte offset, so the registry
    // needs no host-side copy of the materials already resident.
    [[nodiscard]] std::uint32_t append(const std::vector<Material>& materials);

    // Rewinds to empty. Every index handed out before this becomes meaningless,
    // so the scene referencing them must be cleared in the same breath.
    void reset();

    [[nodiscard]] std::uint32_t count() const { return count_; }
    [[nodiscard]] std::uint32_t capacity() const { return capacity_; }

private:
    const Allocator& allocator_;
    const Uploader& uploader_;
    BufferAllocation allocation_;
    VkDeviceSize size_ = 0;
    std::uint32_t capacity_ = 0;
    std::uint32_t count_ = 0;
};

}  // namespace sage::gpu