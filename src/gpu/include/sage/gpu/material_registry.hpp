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
// is written exactly once, at load time, so the upload cost is irrelevant.
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

    // Replaces the whole table. A material's index is its position in the
    // vector, which is what a draw puts in its push constants.
    void upload(const std::vector<Material>& materials);

private:
    const Allocator& allocator_;
    const Uploader& uploader_;
    BufferAllocation allocation_;
    VkDeviceSize size_ = 0;
    std::uint32_t capacity_ = 0;
};

}  // namespace sage::gpu