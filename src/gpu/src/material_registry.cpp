#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/bindless_set.hpp>
#include <sage/gpu/material_registry.hpp>
#include <sage/gpu/uploader.hpp>

namespace sage::gpu {

MaterialRegistry::MaterialRegistry(const Allocator& allocator, const Uploader& uploader,
                                   const BindlessSet& bindless_set, std::uint32_t capacity)
    : allocator_(allocator), uploader_(uploader), capacity_(capacity) {
    SAGE_VERIFY(capacity > 0, "MaterialRegistry: zero capacity");

    size_ = VkDeviceSize{capacity} * sizeof(Material);
    allocation_ = allocator_.create_device_local_buffer(
        size_, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // Registered once, up front: the descriptor names the buffer, not its
    // contents, so it does not need rewriting when the table is filled in.
    bindless_set.write_storage_buffer(k_storage_slot, allocation_.buffer, size_);

    SAGE_LOG_INFO("Material registry: {} slots ({} KiB) at bindless storage slot {}", capacity,
                  size_ / 1024, k_storage_slot);
}

void MaterialRegistry::upload(const std::vector<Material>& materials) {
    SAGE_VERIFY(!materials.empty(), "MaterialRegistry: nothing to upload");
    SAGE_VERIFY(materials.size() <= capacity_, "MaterialRegistry: out of capacity");

    // FRAGMENT_SHADER because the material is read per fragment. Vertex stages
    // never touch it, so widening the destination scope would only over-order.
    uploader_.upload_to_buffer(
        allocation_.buffer, 0, materials.data(), materials.size() * sizeof(Material),
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    SAGE_LOG_INFO("Uploaded {} material(s)", materials.size());
}

MaterialRegistry::~MaterialRegistry() {
    allocator_.destroy_buffer(allocation_);
}

}  // namespace sage::gpu