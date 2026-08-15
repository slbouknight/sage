#include <sage/core/assert.hpp>
#include <sage/gpu/allocator.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/uploader.hpp>
#include <sage/gpu/vk_check.hpp>

#include <cstdint>
#include <cstring>
#include <limits>

namespace sage::gpu {

namespace {
VkCommandPool create_pool(VkDevice device, std::uint32_t family) {
    VkCommandPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    // Every buffer from this pool is recorded once and thrown away.
    info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    info.queueFamilyIndex = family;

    VkCommandPool pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateCommandPool(device, &info, nullptr, &pool));
    return pool;
}

VkCommandBuffer begin_one_shot(VkDevice device, VkCommandPool pool) {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device, &alloc_info, &command_buffer));

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(command_buffer, &begin_info));
    return command_buffer;
}

void submit_and_wait(VkDevice device, VkQueue queue, VkCommandPool pool,
                     VkCommandBuffer command_buffer, VkFence fence) {
    VK_CHECK(vkEndCommandBuffer(command_buffer));

    VkCommandBufferSubmitInfo command_buffer_info{};
    command_buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    command_buffer_info.commandBuffer = command_buffer;

    VkSubmitInfo2 submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &command_buffer_info;

    VK_CHECK(vkQueueSubmit2(queue, 1, &submit, fence));
    VK_CHECK(
        vkWaitForFences(device, 1, &fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max()));
    VK_CHECK(vkResetFences(device, 1, &fence));
    vkFreeCommandBuffers(device, pool, 1, &command_buffer);
}
}  // namespace

Uploader::Uploader(const Allocator& allocator, const Device& device)
    : allocator_(allocator), device_(device) {
    transfer_pool_ = create_pool(device_.handle(), device_.transfer_family());
    graphics_pool_ = create_pool(device_.handle(), device_.graphics_family());

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(device_.handle(), &fence_info, nullptr, &fence_));
}

Uploader::~Uploader() {
    if (fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(device_.handle(), fence_, nullptr);
    }
    if (graphics_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_.handle(), graphics_pool_, nullptr);
    }
    if (transfer_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_.handle(), transfer_pool_, nullptr);
    }
}

void Uploader::upload_to_buffer(VkBuffer dst, VkDeviceSize dst_offset, const void* data,
                                VkDeviceSize size, VkPipelineStageFlags2 dst_stage,
                                VkAccessFlags2 dst_access) const {
    SAGE_VERIFY(size > 0, "Uploader: zero-sized upload");

    const BufferAllocation staging =
        allocator_.create_mapped_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    SAGE_VERIFY(staging.mapped != nullptr, "Uploader: staging buffer is not host-visible");
    std::memcpy(staging.mapped, data, size);
    allocator_.flush(staging, size);

    const std::uint32_t transfer_family = device_.transfer_family();
    const std::uint32_t graphics_family = device_.graphics_family();
    const bool needs_ownership_transfer = transfer_family != graphics_family;

    VkCommandBuffer transfer_cb = begin_one_shot(device_.handle(), transfer_pool_);

    VkBufferCopy2 region{};
    region.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2;
    region.srcOffset = 0;
    region.dstOffset = dst_offset;
    region.size = size;

    VkCopyBufferInfo2 copy_info{};
    copy_info.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2;
    copy_info.srcBuffer = staging.buffer;
    copy_info.dstBuffer = dst;
    copy_info.regionCount = 1;
    copy_info.pRegions = &region;
    vkCmdCopyBuffer2(transfer_cb, &copy_info);

    if (needs_ownership_transfer) {
        VkBufferMemoryBarrier2 release{};
        release.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        release.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        release.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        // A release defines no destination scope -- the matching acquire on the
        // other family supplies it. Setting one here is a spec violation.
        release.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
        release.dstAccessMask = VK_ACCESS_2_NONE;
        release.srcQueueFamilyIndex = transfer_family;
        release.dstQueueFamilyIndex = graphics_family;
        release.buffer = dst;
        release.offset = dst_offset;
        release.size = size;

        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.bufferMemoryBarrierCount = 1;
        dependency.pBufferMemoryBarriers = &release;
        vkCmdPipelineBarrier2(transfer_cb, &dependency);
    }

    submit_and_wait(device_.handle(), device_.transfer_queue(), transfer_pool_, transfer_cb,
                    fence_);

    if (needs_ownership_transfer) {
        VkCommandBuffer graphics_cb = begin_one_shot(device_.handle(), graphics_pool_);

        VkBufferMemoryBarrier2 acquire{};
        acquire.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        // Mirror image of the release: no source scope, only a destination.
        acquire.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        acquire.srcAccessMask = VK_ACCESS_2_NONE;
        acquire.dstStageMask = dst_stage;
        acquire.dstAccessMask = dst_access;
        acquire.srcQueueFamilyIndex = transfer_family;
        acquire.dstQueueFamilyIndex = graphics_family;
        acquire.buffer = dst;
        acquire.offset = dst_offset;
        acquire.size = size;

        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.bufferMemoryBarrierCount = 1;
        dependency.pBufferMemoryBarriers = &acquire;
        vkCmdPipelineBarrier2(graphics_cb, &dependency);

        submit_and_wait(device_.handle(), device_.graphics_queue(), graphics_pool_, graphics_cb,
                        fence_);
    }

    // Safe only now: both submissions have completed.
    allocator_.destroy_buffer(staging);
}

}  // namespace sage::gpu