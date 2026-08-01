#include "application.hpp"

#include <sage/core/log.hpp>
#include <sage/gpu/shader_module.hpp>
#include <sage/gpu/vk_check.hpp>

#include <cstdlib>
#include <filesystem>

namespace sage::app {

namespace {

constexpr std::uint32_t k_initial_width = 1280;
constexpr std::uint32_t k_initial_height = 720;

constexpr VkClearColorValue k_clear_color{{0.05F, 0.05F, 0.07F, 1.0F}};

constexpr VkImageSubresourceRange k_color_range{
    VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};

// The cache belongs in the user's cache dir, not the build tree: it must
// survive `--clean`, and it is machine-specific so it should never be
// committed or copied between machines.
std::filesystem::path pipeline_cache_path() {
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg != nullptr && *xdg != '\0') {
        return std::filesystem::path(xdg) / "sage" / "pipeline_cache.bin";
    }
    if (const char* home = std::getenv("HOME"); home != nullptr && *home != '\0') {
        return std::filesystem::path(home) / ".cache" / "sage" / "pipeline_cache.bin";
    }
    return std::filesystem::path(".sage-pipeline-cache.bin");
}

}  // namespace

Application::Application()
    : window_(k_initial_width, k_initial_height, "sage"),
      instance_("sage", gpu::Window::required_instance_extensions()),
      surface_(instance_, window_),
      physical_device_(gpu::select_physical_device(instance_.handle(), surface_.handle())),
      device_(physical_device_),
      allocator_(instance_, device_),
      swapchain_(device_, surface_.handle(), window_.framebuffer_extent()),
      pipeline_cache_(device_, pipeline_cache_path()),
      pipeline_(device_, std::filesystem::path(SAGE_SHADER_DIR) / "triangle.spv",
                swapchain_.format(), pipeline_cache_.handle()),
      frame_pacer_(device_) {}

Application::~Application() {
    // Everything below must outlive in-flight GPU work.
    device_.wait_idle();
}

void Application::record_triangle(VkCommandBuffer command_buffer, VkImage image,
                                  VkImageView image_view, VkExtent2D extent) const {
    VkImageMemoryBarrier2 to_color{};
    to_color.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_color.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    to_color.srcAccessMask = VK_ACCESS_2_NONE;
    to_color.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    to_color.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    to_color.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_color.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    to_color.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_color.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_color.image = image;
    to_color.subresourceRange = k_color_range;

    VkDependencyInfo to_color_dependency{};
    to_color_dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    to_color_dependency.imageMemoryBarrierCount = 1;
    to_color_dependency.pImageMemoryBarriers = &to_color;
    vkCmdPipelineBarrier2(command_buffer, &to_color_dependency);

    // loadOp CLEAR is what replaces M1's vkCmdClearColorImage.
    VkRenderingAttachmentInfo color_attachment{};
    color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachment.imageView = image_view;
    color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.clearValue.color = k_clear_color;

    VkRenderingInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering.renderArea.offset = {0, 0};
    rendering.renderArea.extent = extent;
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &color_attachment;

    vkCmdBeginRendering(command_buffer, &rendering);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.handle());

    VkViewport viewport{};
    viewport.x = 0.0F;
    viewport.y = 0.0F;
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0F;
    viewport.maxDepth = 1.0F;
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = extent;
    vkCmdSetScissor(command_buffer, 0, 1, &scissor);

    vkCmdDraw(command_buffer, 3, 1, 0, 0);

    vkCmdEndRendering(command_buffer);

    VkImageMemoryBarrier2 to_present{};
    to_present.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_present.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    to_present.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    to_present.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    to_present.dstAccessMask = VK_ACCESS_2_NONE;
    to_present.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
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

        record_triangle(frame.command_buffer, acquired.image, swapchain_.image_view(acquired.index),
                        swapchain_.extent());

        frame_pacer_.submit(device_.graphics_queue(), frame,
                            swapchain_.render_finished(acquired.index));

        const VkResult present_result = swapchain_.present(device_.present_queue(), acquired.index);
        if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR ||
            acquired.result == VK_SUBOPTIMAL_KHR) {
            recreate_swapchain();
        }
    }

    frame_pacer_.wait_all();
    pipeline_cache_.save();
    SAGE_LOG_INFO("Main loop exited");
}

}  // namespace sage::app
