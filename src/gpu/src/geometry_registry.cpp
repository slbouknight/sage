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

VkDeviceSize GeometryRegistry::allocate(VkDeviceSize size, VkDeviceSize alignment) {
    SAGE_VERIFY((alignment & (alignment - 1)) == 0, "Alignment must be a power of two");

    const VkDeviceSize aligned = (head_ + alignment - 1) & ~(alignment - 1);
    SAGE_VERIFY(aligned + size <= capacity_, "GeometryRegistry: out of capacity");

    head_ = aligned + size;
    return aligned;
}

GeometryRegistry::MeshView GeometryRegistry::add_mesh(const void* vertices,
                                                      VkDeviceSize vertex_bytes,
                                                      const std::uint32_t* indices,
                                                      std::uint32_t index_count) {
    SAGE_VERIFY(vertices != nullptr && indices != nullptr, "GeometryRegistry: null mesh data");
    SAGE_VERIFY(vertex_bytes > 0 && index_count > 0, "GeometryRegistry: empty mesh");

    const VkDeviceSize index_bytes = VkDeviceSize{index_count} * sizeof(std::uint32_t);

    const VkDeviceSize vertex_offset = allocate(vertex_bytes, k_vertex_alignment);
    const VkDeviceSize index_offset = allocate(index_bytes, k_index_alignment);

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

GeometryRegistry::~GeometryRegistry() {
    allocator_.destroy_buffer(allocation_);
}

}  // namespace sage::gpu