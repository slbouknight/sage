#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/geometry_registry.hpp>
#include <sage/gpu/uploader.hpp>

namespace sage::gpu {

namespace {
// Conservative: a BDA- addressed vertex struct may contain members that want
// 16 byte alignment, and over-aligning the region start costs a few bytes once.
constexpr VkDeviceSize k_vertex_alignment = 16;

// vkCmdBindIndexBuffer requires the offset be a multiple of the index size.
constexpr VkDeviceSize k_index_alignment = sizeof(std::uint32_t);

constexpr VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

}  // namespace

GeometryRegistry::GeometryRegistry(const Allocator& allocator, const Device& device,
                                   const Uploader& uploader, VkDeviceSize capacity)
    : allocator_(allocator), uploader_(uploader), capacity_(capacity) {
    allocation_ = allocator_.create_device_local_buffer(
        capacity, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    VkBufferDeviceAddressInfo address_info{};
    address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    address_info.buffer = allocation_.buffer;
    base_address_ = vkGetBufferDeviceAddress(device.handle(), &address_info);

    SAGE_LOG_INFO("Geometry registry: {} KiB device-local", capacity / 1024);
}

GeometryRegistry::MeshView GeometryRegistry::add_mesh(const void* vertices,
                                                      VkDeviceSize vertex_bytes,
                                                      const std::uint32_t* indices,
                                                      std::uint32_t index_count) {
    SAGE_VERIFY(vertices != nullptr && indices != nullptr, "GeometryRegistry: null mesh data");
    SAGE_VERIFY(vertex_bytes > 0 && index_count > 0, "GeometryRegistry: empty mesh");

    const VkDeviceSize index_bytes = VkDeviceSize{index_count} * sizeof(std::uint32_t);

    // Both regions are laid out before either is committed. Advancing the head
    // per region would leave the vertices stranded in the buffer when only the
    // indices failed to fit.
    const VkDeviceSize vertex_offset = align_up(head_, k_vertex_alignment);
    const VkDeviceSize index_offset = align_up(vertex_offset + vertex_bytes, k_index_alignment);
    const VkDeviceSize end = index_offset + index_bytes;

    if (end > capacity_) {
        SAGE_LOG_ERROR("Geometry registry full: mesh needs {} KiB, {} KiB of {} KiB free",
                       (end - head_) / 1024, (capacity_ - head_) / 1024, capacity_ / 1024);
        return {};
    }
    head_ = end;

    // The two regions are read by different stages, so each acquire barrier
    // gets the scope that actually applies to it.
    uploader_.upload_to_buffer(allocation_.buffer, vertex_offset, vertices, vertex_bytes,
                               VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                               VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    uploader_.upload_to_buffer(allocation_.buffer, index_offset, indices, index_bytes,
                               VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT, VK_ACCESS_2_INDEX_READ_BIT);

    MeshView view;
    view.vertex_address = base_address_ + vertex_offset;
    view.index_offset = index_offset;
    view.index_count = index_count;
    return view;
}

void GeometryRegistry::reset() {
    // Nothing to free and nothing to clear: the buffer stays, and every byte
    // beyond the head is dead the moment the head moves. Uploads overwrite it
    // in place, so stale contents are never read.
    head_ = 0;
}

GeometryRegistry::~GeometryRegistry() {
    allocator_.destroy_buffer(allocation_);
}

}  // namespace sage::gpu