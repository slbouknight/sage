#include "application.hpp"

#include <sage/core/log.hpp>
#include <sage/core/math.hpp>
#include <sage/gpu/geometry_registry.hpp>
#include <sage/gpu/light.hpp>
#include <sage/gpu/shader_module.hpp>
#include <sage/gpu/vertex.hpp>
#include <sage/gpu/vk_check.hpp>

#include <imgui.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>

namespace sage::app {

namespace {

// Camera setup
const glm::vec3 k_initial_camera_position{2.0F, 1.5F, 3.0F};
constexpr float k_initial_camera_yaw = -2.16F;    // radians
constexpr float k_initial_camera_pitch = -0.39F;  // radians
constexpr float k_field_of_view_degrees = 60.0F;

// Radians per pixel of the mouse movement.
constexpr float k_mouse_sensitivity = 0.003F;

core::CameraInput to_camera_input(const gpu::Window::InputState& input) {
    core::CameraInput camera_input;

    // Only moves the camera while the right button is held; WASD does nothing on its own
    // Returning early keeps that rule in one place/
    if (!input.look_active) {
        return camera_input;
    }

    camera_input.forward = (input.forward ? 1.0F : 0.0F) - (input.back ? 1.0F : 0.0F);
    camera_input.right = (input.right ? 1.0F : 0.0F) - (input.left ? 1.0F : 0.0F);
    camera_input.up = (input.up ? 1.0F : 0.0F) - (input.down ? 1.0F : 0.0F);

    camera_input.yaw_delta = input.cursor_delta_x * k_mouse_sensitivity;
    // Screen Y grows downward; moving the mouse up should pitch  up.
    camera_input.pitch_delta = -input.cursor_delta_y * k_mouse_sensitivity;

    return camera_input;
}

// Must match FrameData in shaders/mesh.slang. Written once per frame and read
// by every draw which is exactly why it is a buffer and not a push constant.
//
// glm's mat4 has alignment 4, not 16, so the alignas is what puts camera_position
// at 64 rather than wherever the compiler feels like. Note the SPIR-V ArrayStride for
// the struct is 76 under scalar layout while sizeof here is 80: the two disagree, which is
// harmless only because a single element is ever dereferenced. Never index a FrameData* as an
// array.
struct FrameData {
    alignas(16) glm::mat4 view_projection{1.0F};
    glm::vec3 camera_position{0.0F};
    std::uint32_t light_count = 0;
    std::array<gpu::Light, gpu::k_max_lights> lights{};
};
static_assert(offsetof(FrameData, view_projection) == 0);
static_assert(offsetof(FrameData, camera_position) == 64);
static_assert(offsetof(FrameData, light_count) == 76);
static_assert(offsetof(FrameData, lights) == 80);
static_assert(sizeof(FrameData) == 464);
// 80 is a multiple of the 16-byte alignment a device address requires, so slot
// N's address is simply base + N * sizeof(FrameData) with no padding.
static_assert(sizeof(FrameData) % 16 == 0);

// Room for a few hundred materials; a real scene revisits this alongside
// k_geometry_capacity, which has the same fixed-size problem.
constexpr std::uint32_t k_max_materials = 256;

// Must match shaders/mesh.slang's PushConstants exactly. The static_asserts
// below turn a layout mismatch into a build failure instead of a GPU fault.
struct PushConstants {
    VkDeviceAddress vertex_address = 0;
    VkDeviceAddress frame_address = 0;
    alignas(16) glm::mat4 model{1.0F};
    std::uint32_t material_index = 0;
};
static_assert(offsetof(PushConstants, vertex_address) == 0);
static_assert(offsetof(PushConstants, frame_address) == 8);
static_assert(offsetof(PushConstants, model) == 16);
static_assert(offsetof(PushConstants, material_index) == 80);
static_assert(sizeof(PushConstants) <= gpu::GraphicsPipeline::k_push_constant_size);

constexpr std::uint32_t k_initial_width = 1280;
constexpr std::uint32_t k_initial_height = 720;

constexpr VkClearColorValue k_clear_color{{0.0036F, 0.0036F, 0.0036F, 1.0F}};

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

Application::Application(const std::filesystem::path& model_path)
    : camera_(k_initial_camera_position, k_initial_camera_yaw, k_initial_camera_pitch),
      window_(k_initial_width, k_initial_height, "sage"),
      instance_("sage", gpu::Window::required_instance_extensions()),
      surface_(instance_, window_),
      physical_device_(gpu::select_physical_device(instance_.handle(), surface_.handle())),
      device_(physical_device_),
      allocator_(instance_, device_),
      bindless_set_(device_),
      uploader_(allocator_, device_),
      geometry_registry_(allocator_, device_, uploader_, k_geometry_capacity),
      sampler_(device_),
      texture_registry_(allocator_, device_, uploader_, bindless_set_, sampler_),
      material_registry_(allocator_, uploader_, bindless_set_, k_max_materials),
      frame_buffer_(allocator_, device_, sizeof(FrameData) * gpu::FramePacer::k_frames_in_flight,
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT),
      swapchain_(device_, surface_.handle(), window_.framebuffer_extent()),
      depth_buffer_(allocator_, device_, swapchain_.extent()),
      pipeline_cache_(device_, pipeline_cache_path()),
      pipeline_(device_,
                gpu::GraphicsPipelineDesc{
                    .spirv_path = std::filesystem::path(SAGE_SHADER_DIR) / "mesh.spv",
                    .color_format = swapchain_.format(),
                    .depth_format = depth_buffer_.format(),
                    .set_layout = bindless_set_.layout(),
                    .cache = pipeline_cache_.handle(),
                }),
      frame_pacer_(device_),
      imgui_(instance_, device_, window_, swapchain_.format(), swapchain_.image_count()) {
    const gpu::LoadedScene loaded =
        gpu::load_gltf(model_path, geometry_registry_, texture_registry_, scene_graph_);

    material_registry_.upload(loaded.materials);

    frame_camera_on(loaded.bounds_min, loaded.bounds_max);
}

Application::~Application() {
    // Everything below must outlive in-flight GPU work.
    device_.wait_idle();
}

void Application::record_scene(VkCommandBuffer command_buffer, VkImage image,
                               VkImageView image_view, VkExtent2D extent,
                               std::uint32_t frame_slot) const {
    VkImageMemoryBarrier2 to_color{};
    to_color.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_color.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
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
    to_depth.srcStageMask =
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    to_depth.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    to_depth.dstStageMask =
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
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

    const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    const glm::mat4 projection =
        core::perspective_vk(glm::radians(k_field_of_view_degrees), aspect, 0.1F, 1000.0F);

    // One write per frame, into this frame's own slot. Writing a single shared
    // slot would race the GPU, which may still be reading the previous frame's
    // copy. begin_frame() has already waited out the work that used this slot.
    FrameData frame_data;
    frame_data.view_projection = projection * camera_.view_matrix();
    frame_data.camera_position = camera_.position();

    // A scene owned light list arrives with the editor. What's important here
    // is that the shader reads data rather than constants, so moving a light
    // is a value change and not a recompile.
    frame_data.light_count = 2;

    frame_data.lights[0].type = gpu::LightType::directional;
    frame_data.lights[0].direction = glm::normalize(glm::vec3(-0.5F, -1.0F, -0.8F));
    frame_data.lights[0].color = glm::vec3(1.0F, 0.96F, 0.9F);
    frame_data.lights[0].intensity = 2.0F;

    // Parked near the lantern head so the falloff is visible against the post.
    frame_data.lights[1].type = gpu::LightType::point;
    frame_data.lights[1].position = glm::vec3(9.6F, 18.0F, 0.0F);
    frame_data.lights[1].color = glm::vec3(1.0F, 0.7F, 0.35F);
    frame_data.lights[1].intensity = 60.0F;
    frame_data.lights[1].range = 12.0F;

    const VkDeviceSize frame_offset = VkDeviceSize{frame_slot} * sizeof(FrameData);
    frame_buffer_.write(&frame_data, sizeof(frame_data), frame_offset);
    const VkDeviceAddress frame_address = frame_buffer_.device_address() + frame_offset;

    for (const gpu::SceneNode& node : scene_graph_.nodes()) {
        if (!node.has_mesh) {
            // Pure transform nodes: glTF hierarchy nodes, and the per-load root.
            continue;
        }

        PushConstants push{};
        push.vertex_address = node.mesh.vertex_address;
        push.frame_address = frame_address;
        push.model = node.world_transform;
        push.material_index = node.material_index;
        vkCmdPushConstants(command_buffer, pipeline_.layout(), VK_SHADER_STAGE_ALL, 0, sizeof(push),
                           &push);
        vkCmdBindIndexBuffer(command_buffer, geometry_registry_.buffer(), node.mesh.index_offset,
                             VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(command_buffer, node.mesh.index_count, 1, 0, 0, 0);
    }

    vkCmdEndRendering(command_buffer);
}

void Application::transition_to_present(VkCommandBuffer command_buffer, VkImage image) {
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

void Application::draw_ui() {
    const ImGuiIO& io = ImGui::GetIO();

    ImGui::Begin("sage");
    ImGui::Text("%.1f fps (%.2f ms)", static_cast<double>(io.Framerate),
                1000.0 / static_cast<double>(io.Framerate));
    ImGui::Separator();
    ImGui::Text("Scene: %zu nodes", scene_graph_.size());
    const glm::vec3 position = camera_.position();
    ImGui::Text("Camera: %.1f, %.1f, %.1f", static_cast<double>(position.x),
                static_cast<double>(position.y), static_cast<double>(position.z));
    ImGui::End();
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

void Application::frame_camera_on(const glm::vec3& bounds_min, const glm::vec3& bounds_max) {
    const glm::vec3 center = (bounds_min + bounds_max) * 0.5F;
    const float radius = glm::length(bounds_max - bounds_min) * 0.5F;

    // Distance at which a sphere of `radius` fills the vertical FOV
    // with margin so the model does not touch the edges of the window.
    const float distance = (radius / std::tan(glm::radians(k_field_of_view_degrees) * 0.5F)) * 1.5F;

    // Offset diagonally so three faces of anything box-like are visible, rather
    // than looking straight down an axis at a flat silhouette.
    const glm::vec3 direction = glm::normalize(glm::vec3(0.6F, 0.4F, 1.0F));
    const glm::vec3 position = center + direction * distance;

    const glm::vec3 to_center = glm::normalize(center - position);
    const float yaw = std::atan2(to_center.z, to_center.x);
    const float pitch = std::asin(to_center.y);

    camera_ = core::Camera(position, yaw, pitch);
    camera_.adjust_speed(std::log(std::max(radius, 0.1F)) / std::log(1.15F));

    SAGE_LOG_INFO("Framed camera at ({:.2f}, {:.2f}, {:.2f}), scene radius {:.2f}", position.x,
                  position.y, position.z, radius);
}

void Application::run() {
    SAGE_LOG_INFO("Entering main loop");

    auto last_frame_time = std::chrono::steady_clock::now();
    SAGE_LOG_INFO("Camera: hold RMB to look, WASD to move, E/Q up/down, scroll to change speed");

    while (!window_.should_close()) {
        gpu::Window::poll_events();

        const auto now = std::chrono::steady_clock::now();
        const float delta_seconds = std::chrono::duration<float>(now - last_frame_time).count();
        last_frame_time = now;

        const gpu::Window::InputState input = window_.sample_input();
        // A panel under the pointer takes precedence, so dragging a slider does
        // not also spin the view.
        const bool ui_has_pointer = gpu::ImGuiLayer::wants_mouse();
        if (!ui_has_pointer && input.look_active && input.scroll_delta != 0.0F) {
            camera_.adjust_speed(input.scroll_delta);
        }
        camera_.update(ui_has_pointer ? core::CameraInput{} : to_camera_input(input),
                       delta_seconds);

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

        // Started only once the frame is certain to be submitted: the
        // out-of-date path above bails without rendering, and an unterminated
        // ImGui frame would trip the next NewFrame().
        gpu::ImGuiLayer::begin_frame();
        draw_ui();

        record_scene(frame.command_buffer, acquired.image, swapchain_.image_view(acquired.index),
                     swapchain_.extent(), frame.slot);
        imgui_.render(frame.command_buffer, swapchain_.image_view(acquired.index),
                      swapchain_.extent());
        transition_to_present(frame.command_buffer, acquired.image);

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
