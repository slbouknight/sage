#include "application.hpp"

#include <sage/gpu/geometry_registry.hpp>
#include <sage/core/log.hpp>
#include <sage/core/math.hpp>
#include <sage/gpu/shader_module.hpp>
#include <sage/gpu/vk_check.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>

namespace sage::app {

namespace {

// Must match shaders/triangle.slang's PushConstants exactly. The static_asserts
// below turn a layout mismatch into a build failure instead of a GPU fault.
struct PushConstants {
    VkDeviceAddress vertex_address = 0;
    alignas(16) glm::mat4 mvp{1.0F};
};
static_assert(offsetof(PushConstants, vertex_address) == 0);
static_assert(offsetof(PushConstants, mvp) == 16);
static_assert(sizeof(PushConstants) <= gpu::GraphicsPipeline::k_push_constant_size);

// Must match Vertex in shaders/triangle.slang. Slang's "natural" layout for
// BDA-accessed structs is C-like packing, no std430 padding, which is exactly
// what glm::vec3's give: ArrayStride 24, offsets 0 and 12.
struct Vertex
{
    glm::vec3 position;
    glm::vec3 color;
};
static_assert(sizeof(Vertex) == 24);
static_assert(offsetof(Vertex, position) == 0);
static_assert(offsetof(Vertex, color) == 12);

// 24 vertices, not 8: each face needs its own color, so corners are
// duplicated per face. Winding is CCW viewed from outside.
const std::array<Vertex, 24> k_cube_vertices{{
    // +Z front, red
    {{-0.5F, -0.5F, 0.5F}, {1.0F, 0.0F, 0.0F}},
    {{0.5F, -0.5F, 0.5F}, {1.0F, 0.0F, 0.0F}},
    {{0.5F, 0.5F, 0.5F}, {1.0F, 0.0F, 0.0F}},
    {{-0.5F, 0.5F, 0.5F}, {1.0F, 0.0F, 0.0F}},
    // -Z back, cyan
    {{0.5F, -0.5F, -0.5F}, {0.0F, 1.0F, 1.0F}},
    {{-0.5F, -0.5F, -0.5F}, {0.0F, 1.0F, 1.0F}},
    {{-0.5F, 0.5F, -0.5F}, {0.0F, 1.0F, 1.0F}},
    {{0.5F, 0.5F, -0.5F}, {0.0F, 1.0F, 1.0F}},
    // +X right, green
    {{0.5F, -0.5F, 0.5F}, {0.0F, 1.0F, 0.0F}},
    {{0.5F, -0.5F, -0.5F}, {0.0F, 1.0F, 0.0F}},
    {{0.5F, 0.5F, -0.5F}, {0.0F, 1.0F, 0.0F}},
    {{0.5F, 0.5F, 0.5F}, {0.0F, 1.0F, 0.0F}},
    // -X left, magenta
    {{-0.5F, -0.5F, -0.5F}, {1.0F, 0.0F, 1.0F}},
    {{-0.5F, -0.5F, 0.5F}, {1.0F, 0.0F, 1.0F}},
    {{-0.5F, 0.5F, 0.5F}, {1.0F, 0.0F, 1.0F}},
    {{-0.5F, 0.5F, -0.5F}, {1.0F, 0.0F, 1.0F}},
    // +Y top, blue
    {{-0.5F, 0.5F, 0.5F}, {0.0F, 0.0F, 1.0F}},
    {{0.5F, 0.5F, 0.5F}, {0.0F, 0.0F, 1.0F}},
    {{0.5F, 0.5F, -0.5F}, {0.0F, 0.0F, 1.0F}},
    {{-0.5F, 0.5F, -0.5F}, {0.0F, 0.0F, 1.0F}},
    // -Y bottom, yellow
    {{-0.5F, -0.5F, -0.5F}, {1.0F, 1.0F, 0.0F}},
    {{0.5F, -0.5F, -0.5F}, {1.0F, 1.0F, 0.0F}},
    {{0.5F, -0.5F, 0.5F}, {1.0F, 1.0F, 0.0F}},
    {{-0.5F, -0.5F, 0.5F}, {1.0F, 1.0F, 0.0F}},
}};

// Two triangles per face, from each face's four consecutive vertices.
const std::array<std::uint32_t, 36> k_cube_indices{
    0,  1,  2,  0,  2,  3,   // +Z
    4,  5,  6,  4,  6,  7,   // -Z
    8,  9,  10, 8,  10, 11,  // +X
    12, 13, 14, 12, 14, 15,  // -X
    16, 17, 18, 16, 18, 19,  // +Y
    20, 21, 22, 20, 22, 23,  // -Y
};

constexpr std::uint32_t k_initial_width = 1280;
constexpr std::uint32_t k_initial_height = 720;

constexpr VkClearColorValue k_clear_color{{0.05F, 0.05F, 0.07F, 1.0F}};

constexpr VkImageSubresourceRange k_color_range{
    VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};

constexpr VkImageSubresourceRange k_depth_range{
    VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};

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

// 4MiB: far more than a cube needs, and a round number to revisit when adding real meshes later
constexpr VkDeviceSize k_geometry_capacity = 4ULL * 1024 * 1024;
}  // namespace

Application::Application()
    : window_(k_initial_width, k_initial_height, "sage"),
      instance_("sage", gpu::Window::required_instance_extensions()),
      surface_(instance_, window_),
      physical_device_(gpu::select_physical_device(instance_.handle(), surface_.handle())),
      device_(physical_device_),
      allocator_(instance_, device_),
      bindless_set_(device_),
      uploader_(allocator_, device_),
      geometry_registry_(allocator_, device_, uploader_, k_geometry_capacity),
      swapchain_(device_, surface_.handle(), window_.framebuffer_extent()),
      depth_buffer_(allocator_, device_, swapchain_.extent()),
      pipeline_cache_(device_, pipeline_cache_path()),
      pipeline_(device_,
                gpu::GraphicsPipelineDesc{
                    .spirv_path = std::filesystem::path(SAGE_SHADER_DIR) / "triangle.spv",
                    .color_format = swapchain_.format(),
                    .depth_format = depth_buffer_.format(),
                    .set_layout = bindless_set_.layout(),
                    .cache = pipeline_cache_.handle(),
                }),
      frame_pacer_(device_)
      {
        cube_ = geometry_registry_.add_mesh(k_cube_vertices.data(),
                                            sizeof(Vertex) * k_cube_vertices.size(),
                                            k_cube_indices.data(),
                                            static_cast<std::uint32_t>(k_cube_indices.size()));
       }

Application::~Application() {
    // Everything below must outlive in-flight GPU work.
    device_.wait_idle();
}

void Application::record_cube(VkCommandBuffer command_buffer, VkImage image,
                              VkImageView image_view, VkExtent2D extent, float time_seconds) const {
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

    VkImageMemoryBarrier2 to_depth{};
    to_depth.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_depth.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    to_depth.srcAccessMask = VK_ACCESS_2_NONE;
    to_depth.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    to_depth.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    to_depth.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_depth.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    to_depth.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_depth.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_depth.image = depth_buffer_.image();
    to_depth.subresourceRange = k_depth_range;
    
    const std::array<VkImageMemoryBarrier2, 2> begin_barriers{to_color, to_depth};

    VkDependencyInfo begin_dependency{};
    begin_dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    begin_dependency.imageMemoryBarrierCount = static_cast<std::uint32_t>(begin_barriers.size());
    begin_dependency.pImageMemoryBarriers = begin_barriers.data();
    vkCmdPipelineBarrier2(command_buffer, &begin_dependency);

    // loadOp CLEAR is what replaces M1's vkCmdClearColorImage.
    VkRenderingAttachmentInfo color_attachment{};
    color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachment.imageView = image_view;
    color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.clearValue.color = k_clear_color;

    VkRenderingAttachmentInfo depth_attachment{};
    depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_attachment.imageView = depth_buffer_.view();
    depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.clearValue.depthStencil = {1.0F, 0};

    VkRenderingInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering.renderArea.offset = {0, 0};
    rendering.renderArea.extent = extent;
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &color_attachment;
    rendering.pDepthAttachment = &depth_attachment;

    vkCmdBeginRendering(command_buffer, &rendering);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.handle());

    const VkDescriptorSet descriptor_set = bindless_set_.handle();
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.layout(), 0,
                            1, &descriptor_set, 0, nullptr);
    
    const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    const glm::mat4 projection = core::perspective_vk(glm::radians(60.0F), aspect, 0.1F, 100.0F);
    const glm::mat4 view = glm::lookAt(glm::vec3(2.0F, 1.5F, 3.0F),   // eye
                                       glm::vec3(0.0F, 0.0F, 0.0F),   // target
                                       glm::vec3(0.0F, 1.0F, 0.0F));  // up
    const glm::mat4 model = glm::rotate(glm::mat4(1.0F), time_seconds * 0.5F,
                                        glm::vec3(0.0F, 1.0F, 0.0F));

    PushConstants push{};
    push.vertex_address = cube_.vertex_address;
    push.mvp = projection * view * model;
    vkCmdPushConstants(command_buffer, pipeline_.layout(), VK_SHADER_STAGE_ALL, 0, sizeof(push),
                       &push);

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

    vkCmdBindIndexBuffer(command_buffer, geometry_registry_.buffer(),
                         cube_.index_offset, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(command_buffer, cube_.index_count, 1, 0, 0, 0);

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
    depth_buffer_.recreate(swapchain_.extent());
    return true;
}

void Application::run() {
    SAGE_LOG_INFO("Entering main loop");

    const auto start_time = std::chrono::steady_clock::now();

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

        const float time_seconds = 
        std::chrono::duration<float>(std::chrono::steady_clock::now() -
        start_time).count();
        
        record_cube(frame.command_buffer, acquired.image, swapchain_.image_view(acquired.index),
                        swapchain_.extent(), time_seconds);

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
