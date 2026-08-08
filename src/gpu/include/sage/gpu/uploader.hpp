#pragma once

#include <vulkan/vulkan.h>

namespace sage::gpu
{

class Allocator;
class Device;

// One-shot, fully synchronous buffer uploads: staging buffer -> transfer-queue
// copy -> queue family ownership transfer to graphics. Every call blocks until
// the data is safe for a graphics submission to read.
//
// Startup-only by design. Streaming uploads mid-frame would need a ring of
// staging buffers and semaphore-based synchronisation rather than fence waits;
// M3 does not need that, and building it now would be speculative.
class Uploader {
public:
    Uploader(const Allocator& allocator, const Device& device);
    ~Uploader();

    Uploader(const Uploader&) = delete;
    Uploader& operator=(const Uploader&) = delete;
    Uploader(Uploader&&) = delete;
    Uploader& operator=(Uploader&&) = delete;

    // Copies `size` bytes from `data` into `dst` at `dst_offset`. Returns once
    // the copy has completed and -- when the transfer and graphics families
    // differ -- ownership has been acquired by the graphics family.
    //
    // dst_stage/dst_access describe how the graphics queue will first read this
    // data; they form the acquire barrier's destination scope.
    void upload_to_buffer(VkBuffer dst, VkDeviceSize dst_offset, const void* data,
                          VkDeviceSize size, VkPipelineStageFlags2 dst_stage,
                          VkAccessFlags2 dst_access) const;

private:
    const Allocator& allocator_;
    const Device& device_;

    VkCommandPool transfer_pool_ = VK_NULL_HANDLE;
    VkCommandPool graphics_pool_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
};

} // namespace sage::gpu