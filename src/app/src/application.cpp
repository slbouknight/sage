#include "application.hpp"

#include <sage/core/log.hpp>
#include <sage/gpu/vk_check.hpp>

namespace sage::app {

namespace {

constexpr std::uint32_t k_initial_width = 1280;
constexpr std::uint32_t k_initial_height = 720;

constexpr VkClearColorValue k_clear_color{{0.05F, 0.05F, 0.07F, 1.0F}};

constexpr VkImageSubresourceRange k_color_range{
    VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};

}  // namespace

Application::Application()
    : window_(k_initial_width, k_initial_height, "sage"),
      instance_("sage", gpu::Window::required_instance_extensions()),
      surface_(instance_, window_),
      physical_device_(gpu::select_physical_device(instance_.handle(), surface_.handle())),
      device_(physical_device_),
      allocator_(instance_, device_),
      swapchain_(device_, surface_.handle(), window_.framebuffer_extent()),
      frame_pacer_(device_) {}

Application::~Application() {
    // Everything below must outlive in-flight GPU work.
    device_.wait_idle();
}

void Application::record_clear(VkCommandBuffer command_buffer, VkImage image) const {
    // The image contents from the previous present are not needed, so the
    // transition uses UNDEFINED as the old layout and discards them.
    VkImageMemoryBarrier2 to_transfer{};
    to_transfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_transfer.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    to_transfer.srcAccessMask = VK_ACCESS_2_NONE;
    to_transfer.dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    to_transfer.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    to_transfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_transfer.image = image;
    to_transfer.subresourceRange = k_color_range;

    VkDependencyInfo to_transfer_dependency{};
    to_transfer_dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    to_transfer_dependency.imageMemoryBarrierCount = 1;
    to_transfer_dependency.pImageMemoryBarriers = &to_transfer;
    vkCmdPipelineBarrier2(command_buffer, &to_transfer_dependency);

    vkCmdClearColorImage(command_buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         &k_clear_color, 1, &k_color_range);

    VkImageMemoryBarrier2 to_present{};
    to_present.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_present.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    to_present.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    to_present.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    to_present.dstAccessMask = VK_ACCESS_2_NONE;
    to_present.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    to_present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_present.image = image;
    to_present.subresourceRange = k_color_range;

    VkDependencyInfo to_present_dependency{};
    to_present_dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    to_present_dependency.imageMemoryBarrierCount = 1;
    to_present_dependency.pImageMemoryBarriers = &to_present;
    vkCmdPipelineBarrier2(command_buffer, &to_present_dependency);
}

bool Application::recreate_swapchain() {
    VkExtent2D extent = window_.framebuffer_extent();
    if (extent.width == 0 || extent.height == 0) {
        return false;
    }
    swapchain_.recreate(extent);
    return true;
}

void Application::run() {
    SAGE_LOG_INFO("Entering main loop");

    while (!window_.should_close()) {
        gpu::Window::poll_events();

        const VkExtent2D extent = window_.framebuffer_extent();
        if (extent.width == 0 || extent.height == 0) {
            // Minimized: idle instead of spinning, and never build a
            // zero-extent swapchain.
            gpu::Window::wait_events();
            continue;
        }

        if (window_.consume_resized()) {
            if (!recreate_swapchain()) {
                continue;
            }
        }

        const gpu::FrameContext frame = frame_pacer_.begin_frame();

        const gpu::AcquiredImage acquired = swapchain_.acquire(frame.image_available);
        if (acquired.result == VK_ERROR_OUT_OF_DATE_KHR) {
            // The command buffer was begun but never submitted; the next
            // begin_frame() resets this slot's pool, and image_available was
            // not signalled, so nothing leaks.
            VK_CHECK(vkEndCommandBuffer(frame.command_buffer));
            recreate_swapchain();
            continue;
        }

        record_clear(frame.command_buffer, acquired.image);

        frame_pacer_.submit(device_.graphics_queue(), frame,
                            swapchain_.render_finished(acquired.index));

        const VkResult present_result = swapchain_.present(device_.present_queue(), acquired.index);
        if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR ||
            acquired.result == VK_SUBOPTIMAL_KHR) {
            recreate_swapchain();
        }
    }

    frame_pacer_.wait_all();
    SAGE_LOG_INFO("Main loop exited");
}

}  // namespace sage::app
