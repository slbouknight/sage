#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>

namespace sage::gpu {

class Device;

struct FrameContext {
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkSemaphore image_available = VK_NULL_HANDLE;
    std::uint32_t slot = 0;
};

// CPU/GPU pacing runs on one timeline semaphore; the binary semaphores exist
// only because the WSI acquire/present entry points require them. See ADR 0006.
class FramePacer {
public:
    static constexpr std::uint32_t k_frames_in_flight = 2;

    explicit FramePacer(const Device& device);
    ~FramePacer();

    FramePacer(const FramePacer&) = delete;
    FramePacer& operator=(const FramePacer&) = delete;
    FramePacer(FramePacer&&) = delete;
    FramePacer& operator=(FramePacer&&) = delete;

    // Blocks until the GPU is done with this slot's command buffer, then
    // resets and begins recording.
    [[nodiscard]] FrameContext begin_frame();

    // Ends recording and submits, signalling both the timeline and the given
    // per-image binary semaphore.
    void submit(VkQueue queue, const FrameContext& frame, VkSemaphore render_finished);

    // Waits for all in-flight work. Used before teardown.
    void wait_all() const;

private:
    const Device& device_;

    VkSemaphore timeline_ = VK_NULL_HANDLE;
    std::uint64_t frame_counter_ = 0;

    std::array<VkCommandPool, k_frames_in_flight> command_pools_{};
    std::array<VkCommandBuffer, k_frames_in_flight> command_buffers_{};
    std::array<VkSemaphore, k_frames_in_flight> image_available_{};
    // Timeline value each slot's last submission signalled.
    std::array<std::uint64_t, k_frames_in_flight> slot_values_{};
};

}  // namespace sage::gpu
