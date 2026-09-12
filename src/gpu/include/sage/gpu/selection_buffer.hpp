#pragma once

#include <sage/gpu/allocator.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <span>

namespace sage::gpu {

class BindlessSet;
class Uploader;

// One flag per scene node, telling the outline pass whether that node is part
// of the selection.
//
// A single selected id in a push constant was enough while a click selected one
// mesh. Selecting a node now means selecting everything beneath it, and a
// subtree is not a contiguous range of indices -- SceneGraph guarantees only
// that parents precede children, not that descendants are adjacent. Testing
// membership per pixel therefore needs the answer precomputed per node, which
// is what this holds.
class SelectionBuffer {
public:
    // Slot in BindlessSet's storage-buffer array; 0 is the material table.
    static constexpr std::uint32_t k_storage_slot = 1;

    // 256 KiB of device memory. Far past any scene the geometry budget allows,
    // and a flag array is the cheapest thing here to over-provision.
    static constexpr std::uint32_t k_max_nodes = 65536;

    SelectionBuffer(const Allocator& allocator, const Uploader& uploader,
                    const BindlessSet& bindless_set);
    ~SelectionBuffer();

    SelectionBuffer(const SelectionBuffer&) = delete;
    SelectionBuffer& operator=(const SelectionBuffer&) = delete;
    SelectionBuffer(SelectionBuffer&&) = delete;
    SelectionBuffer& operator=(SelectionBuffer&&) = delete;

    // Replaces the flags. Blocking, which is why this is called when the
    // selection changes rather than every frame. An empty span is a no-op: the
    // outline pass is skipped entirely when nothing is selected, so there is
    // nothing to clear on the device.
    void update(std::span<const std::uint32_t> flags);

private:
    const Allocator& allocator_;
    const Uploader& uploader_;
    BufferAllocation allocation_;
    VkDeviceSize size_ = 0;
};

}  // namespace sage::gpu
