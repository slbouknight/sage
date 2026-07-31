#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace sage::gpu {

class Device;

struct AcquiredImage {
    VkResult result = VK_SUCCESS;
    std::uint32_t index = 0;
    VkImage image = VK_NULL_HANDLE;
};

// No VkImageViews here: vkCmdClearColorImage works on the image directly, and
// views are attachment plumbing that arrives with M2's dynamic rendering.
class Swapchain {
public:
    Swapchain(const Device& device, VkSurfaceKHR surface, VkExtent2D extent);
    ~Swapchain();

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    Swapchain(Swapchain&&) = delete;
    Swapchain& operator=(Swapchain&&) = delete;

    void recreate(VkExtent2D extent);

    // VK_ERROR_OUT_OF_DATE_KHR / VK_SUBOPTIMAL_KHR are returned, not asserted:
    // the caller recreates in response.
    [[nodiscard]] AcquiredImage acquire(VkSemaphore image_available);
    [[nodiscard]] VkResult present(VkQueue queue, std::uint32_t image_index);

    [[nodiscard]] VkSemaphore render_finished(std::uint32_t image_index) const {
        return render_finished_[image_index];
    }

    [[nodiscard]] VkExtent2D extent() const { return extent_; }
    [[nodiscard]] VkFormat format() const { return format_; }

private:
    void create(VkExtent2D extent);
    void destroy();

    const Device& device_;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> images_;
    // One per swapchain image, not per frame-in-flight: the presentation
    // engine may still be consuming an older frame's semaphore when a
    // frame-in-flight slot is reused. See ADR 0006.
    std::vector<VkSemaphore> render_finished_;

    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR color_space_ = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkPresentModeKHR present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
    VkExtent2D extent_{};
};

}  // namespace sage::gpu
