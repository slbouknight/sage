#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/bindless_set.hpp>
#include <sage/gpu/selection_buffer.hpp>
#include <sage/gpu/uploader.hpp>

namespace sage::gpu {

SelectionBuffer::SelectionBuffer(const Allocator& allocator, const Uploader& uploader,
                                 const BindlessSet& bindless_set)
    : allocator_(allocator), uploader_(uploader) {
    size_ = VkDeviceSize{k_max_nodes} * sizeof(std::uint32_t);
    allocation_ = allocator_.create_device_local_buffer(
        size_, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // Device-local rather than host-visible for the same reason as the material
    // table: the outline pass reads this once per covered pixel, and a
    // host-visible buffer would pull every one of those reads across the bus.
    bindless_set.write_storage_buffer(k_storage_slot, allocation_.buffer, size_);

    SAGE_LOG_INFO("Selection buffer: {} node flags ({} KiB) at bindless storage slot {}",
                  k_max_nodes, size_ / 1024, k_storage_slot);
}

void SelectionBuffer::update(std::span<const std::uint32_t> flags) {
    if (flags.empty()) {
        return;
    }
    SAGE_VERIFY(flags.size() <= k_max_nodes, "SelectionBuffer: scene exceeds flag capacity");

    uploader_.upload_to_buffer(
        allocation_.buffer, 0, flags.data(), flags.size() * sizeof(std::uint32_t),
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
}

SelectionBuffer::~SelectionBuffer() {
    allocator_.destroy_buffer(allocation_);
}

}  // namespace sage::gpu
