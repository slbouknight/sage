#include <sage/core/log.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/frame_pacer.hpp>
#include <sage/gpu/vk_check.hpp>

#include <limits>

namespace sage::gpu {

FramePacer::FramePacer(const Device& device) : device_(device) {
    VkSemaphoreTypeCreateInfo type_info{};
    type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type_info.initialValue = 0;

    VkSemaphoreCreateInfo timeline_info{};
    timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    timeline_info.pNext = &type_info;
    VK_CHECK(vkCreateSemaphore(device_.handle(), &timeline_info, nullptr, &timeline_));

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = device_.graphics_family();
    // Reset the whole pool per frame rather than individual buffers.
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

    VkSemaphoreCreateInfo binary_info{};
    binary_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    for (std::uint32_t i = 0; i < k_frames_in_flight; ++i) {
        VK_CHECK(vkCreateCommandPool(device_.handle(), &pool_info, nullptr, &command_pools_[i]));

        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = command_pools_[i];
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device_.handle(), &alloc_info, &command_buffers_[i]));

        VK_CHECK(vkCreateSemaphore(device_.handle(), &binary_info, nullptr, &image_available_[i]));

        slot_values_[i] = 0;
    }
}

FramePacer::~FramePacer() {
    for (std::uint32_t i = 0; i < k_frames_in_flight; ++i) {
        if (image_available_[i] != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_.handle(), image_available_[i], nullptr);
        }
        if (command_pools_[i] != VK_NULL_HANDLE) {
            // Frees the pool's command buffers too.
            vkDestroyCommandPool(device_.handle(), command_pools_[i], nullptr);
        }
    }
    if (timeline_ != VK_NULL_HANDLE) {
        vkDestroySemaphore(device_.handle(), timeline_, nullptr);
    }
}

FrameContext FramePacer::begin_frame() {
    const std::uint32_t slot = static_cast<std::uint32_t>(frame_counter_ % k_frames_in_flight);

    // Wait until this slot's previous submission has completed before reusing
    // its command pool.
    if (slot_values_[slot] != 0) {
        VkSemaphoreWaitInfo wait_info{};
        wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wait_info.semaphoreCount = 1;
        wait_info.pSemaphores = &timeline_;
        wait_info.pValues = &slot_values_[slot];
        VK_CHECK(vkWaitSemaphores(device_.handle(), &wait_info,
                                  std::numeric_limits<std::uint64_t>::max()));
    }

    VK_CHECK(vkResetCommandPool(device_.handle(), command_pools_[slot], 0));

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(command_buffers_[slot], &begin_info));

    return FrameContext{command_buffers_[slot], image_available_[slot], slot};
}

void FramePacer::submit(VkQueue queue, const FrameContext& frame, VkSemaphore render_finished) {
    VK_CHECK(vkEndCommandBuffer(frame.command_buffer));

    const std::uint64_t signal_value = frame_counter_ + 1;

    VkSemaphoreSubmitInfo wait_info{};
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    wait_info.semaphore = frame.image_available;
    wait_info.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkCommandBufferSubmitInfo command_info{};
    command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    command_info.commandBuffer = frame.command_buffer;

    // Binary for the presentation engine, timeline for our own CPU pacing.
    std::array<VkSemaphoreSubmitInfo, 2> signal_infos{};
    signal_infos[0].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signal_infos[0].semaphore = render_finished;
    signal_infos[0].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    signal_infos[1].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signal_infos[1].semaphore = timeline_;
    signal_infos[1].value = signal_value;
    signal_infos[1].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkSubmitInfo2 submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submit.waitSemaphoreInfoCount = 1;
    submit.pWaitSemaphoreInfos = &wait_info;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &command_info;
    submit.signalSemaphoreInfoCount = static_cast<std::uint32_t>(signal_infos.size());
    submit.pSignalSemaphoreInfos = signal_infos.data();

    VK_CHECK(vkQueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE));

    slot_values_[frame.slot] = signal_value;
    frame_counter_ = signal_value;
}

void FramePacer::wait_all() const {
    if (frame_counter_ == 0) {
        return;
    }
    VkSemaphoreWaitInfo wait_info{};
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait_info.semaphoreCount = 1;
    wait_info.pSemaphores = &timeline_;
    wait_info.pValues = &frame_counter_;
    VK_CHECK(
        vkWaitSemaphores(device_.handle(), &wait_info, std::numeric_limits<std::uint64_t>::max()));
}

}  // namespace sage::gpu
