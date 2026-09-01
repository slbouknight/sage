#include <sage/core/assert.hpp>
#include <sage/gpu/allocator.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/uploader.hpp>
#include <sage/gpu/vk_check.hpp>

#include <array>
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

// The layout/range boilerplate every image barrier below repeats. The synchronization scopes
// are deliberately left to the call site: they are the part that differs, and the part
// worth reading.
VkImageMemoryBarrier2 image_barrier(VkImage image, std::uint32_t base_level,
                                    std::uint32_t level_count, VkImageLayout old_layout,
                                    VkImageLayout new_layout) {
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = base_level;
    barrier.subresourceRange.levelCount = level_count;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    return barrier;
}

void record_barriers(VkCommandBuffer command_buffer, const VkImageMemoryBarrier2* barriers,
                     std::uint32_t count) {
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.imageMemoryBarrierCount = count;
    dependency.pImageMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(command_buffer, &dependency);
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

void Uploader::upload_to_image(VkImage image, VkExtent2D extent, std::uint32_t mip_levels,
                               const void* data, VkDeviceSize size) const {
    SAGE_VERIFY(size > 0, "Uploader: zero-sized image upload");
    SAGE_VERIFY(mip_levels > 0, "Uploader: image needs at least one mip level");

    const BufferAllocation staging =
        allocator_.create_mapped_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    SAGE_VERIFY(staging.mapped != nullptr, "Uploader: staging buffer is not host-visible");
    std::memcpy(staging.mapped, data, size);
    allocator_.flush(staging, size);

    VkCommandBuffer command_buffer = begin_one_shot(device_.handle(), graphics_pool_);

    // TOP_OF_PIPE with an empty access mask is the "wait for nothing" idiom:
    // the image was just created, so there is no prior work to order against.
    VkImageMemoryBarrier2 to_transfer_dst = image_barrier(
        image, 0, mip_levels, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    to_transfer_dst.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    to_transfer_dst.srcAccessMask = VK_ACCESS_2_NONE;
    // A layout transition is itself a write, and this one covers every level.
    // COPY alone would leave it unordered against the blits that write levels 1
    // and up -- a write-after-write hazard synchronization validation catches
    // and ordinary validation does not.
    to_transfer_dst.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT;
    to_transfer_dst.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    record_barriers(command_buffer, &to_transfer_dst, 1);

    VkBufferImageCopy2 region{};
    region.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
    region.bufferOffset = 0;
    // Zero means "rows are tightly packed to imageExtent", which is what stb
    // hands back. A non-zero value would describe padding between rows.
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {extent.width, extent.height, 1};

    VkCopyBufferToImageInfo2 copy_info{};
    copy_info.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2;
    copy_info.srcBuffer = staging.buffer;
    copy_info.dstImage = image;
    copy_info.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    copy_info.regionCount = 1;
    copy_info.pRegions = &region;
    vkCmdCopyBufferToImage2(command_buffer, &copy_info);

    // Each level is half of the one above it. Blitting every level straight
    // from level 0 would be wrong, not just slower: LINEAR samples a 2x2
    // neighbourhood regardless of the reduction factor, so a 512 -> 8 blit
    // would read four texels and ignore the other 4092.
    auto mip_width = static_cast<std::int32_t>(extent.width);
    auto mip_height = static_cast<std::int32_t>(extent.height);

    for (std::uint32_t level = 1; level < mip_levels; ++level) {
        VkImageMemoryBarrier2 to_transfer_src =
            image_barrier(image, level - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        // COPY covers level 0 (written by the buffer copy); BLIT covers every
        // level after it (written by the previous iteration's blit).
        to_transfer_src.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT;
        to_transfer_src.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        to_transfer_src.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
        to_transfer_src.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        record_barriers(command_buffer, &to_transfer_src, 1);

        // Clamped at 1: a non-square texture hits 1 on one axis first and must
        // keep halving the other rather than collapsing to zero.
        const std::int32_t next_width = mip_width > 1 ? mip_width / 2 : 1;
        const std::int32_t next_height = mip_height > 1 ? mip_height / 2 : 1;

        VkImageBlit2 blit{};
        blit.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = level - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mip_width, mip_height, 1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = level;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {next_width, next_height, 1};

        VkBlitImageInfo2 blit_info{};
        blit_info.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
        blit_info.srcImage = image;
        blit_info.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        blit_info.dstImage = image;
        blit_info.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        blit_info.regionCount = 1;
        blit_info.pRegions = &blit;
        blit_info.filter = VK_FILTER_LINEAR;
        vkCmdBlitImage2(command_buffer, &blit_info);

        mip_width = next_width;
        mip_height = next_height;
    }

    // Two ranges, because the levels are not in the same layout: everything
    // above the last was flipped to TRANSFER_SRC so the next blit could read
    // it, while the last level was only ever written.
    std::array<VkImageMemoryBarrier2, 2> to_shader_read{};
    std::uint32_t barrier_count = 0;

    if (mip_levels > 1) {
        VkImageMemoryBarrier2 read_levels =
            image_barrier(image, 0, mip_levels - 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        read_levels.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
        read_levels.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        read_levels.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        read_levels.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        to_shader_read[barrier_count] = read_levels;
        ++barrier_count;
    }

    VkImageMemoryBarrier2 last_level =
        image_barrier(image, mip_levels - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    // With a single level nothing blitted, so the write to wait on came from
    // the buffer copy instead.
    last_level.srcStageMask =
        mip_levels > 1 ? VK_PIPELINE_STAGE_2_BLIT_BIT : VK_PIPELINE_STAGE_2_COPY_BIT;
    last_level.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    last_level.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    last_level.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    to_shader_read[barrier_count] = last_level;
    ++barrier_count;

    record_barriers(command_buffer, to_shader_read.data(), barrier_count);

    submit_and_wait(device_.handle(), device_.graphics_queue(), graphics_pool_, command_buffer,
                    fence_);

    allocator_.destroy_buffer(staging);
}

}  // namespace sage::gpu