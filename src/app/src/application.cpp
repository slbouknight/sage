#include "application.hpp"

#include <sage/core/log.hpp>
#include <sage/core/math.hpp>
#include <sage/gpu/geometry_registry.hpp>
#include <sage/gpu/light.hpp>
#include <sage/gpu/screenshot.hpp>
#include <sage/gpu/selection_buffer.hpp>
#include <sage/gpu/shader_module.hpp>
#include <sage/gpu/vertex.hpp>
#include <sage/gpu/vk_check.hpp>

#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
// The DockBuilder API is internal and has no public equivalent. Confined to
// this file, and only for the one-time default layout; everything else uses
// the public header.
#include <ImGuizmo.h>
#include <imgui_internal.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace sage::app {

namespace {

// Camera setup
const glm::vec3 k_initial_camera_position{2.0F, 1.5F, 3.0F};
constexpr float k_initial_camera_yaw = -2.16F;    // radians
constexpr float k_initial_camera_pitch = -0.39F;  // radians
constexpr float k_field_of_view_degrees = 60.0F;
constexpr float k_near_plane = 0.1F;
constexpr float k_far_plane = 1000.0F;

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

// Materials are 64 bytes, so this is 64 KiB of device memory for the whole
// table. Cheap enough that overrunning it is a reason to raise the number
// rather than a case MaterialRegistry has to degrade around.
constexpr std::uint32_t k_max_materials = 1024;

// Must match shaders/mesh.slang's PushConstants exactly. The static_asserts
// below turn a layout mismatch into a build failure instead of a GPU fault.
struct PushConstants {
    VkDeviceAddress vertex_address = 0;
    VkDeviceAddress frame_address = 0;
    alignas(16) glm::mat4 model{1.0F};
    std::uint32_t material_index = 0;
    std::uint32_t object_id = 0;
};
static_assert(offsetof(PushConstants, vertex_address) == 0);
static_assert(offsetof(PushConstants, frame_address) == 8);
static_assert(offsetof(PushConstants, model) == 16);
static_assert(offsetof(PushConstants, material_index) == 80);
// Verified against the compiled SPIR-V, which decorates this member Offset 84.
static_assert(offsetof(PushConstants, object_id) == 84);
static_assert(sizeof(PushConstants) <= gpu::GraphicsPipeline::k_push_constant_size);

// Must match PushConstants in shaders/outline.slang. The padding is not
// cosmetic: Slang aligns an int2 to 8 and a float3 to 16, which is what puts
// these members where the static_asserts say they are.
struct OutlinePushConstants {
    glm::ivec2 viewport_origin{0, 0};
    glm::ivec2 viewport_size{0, 0};
    alignas(16) glm::vec3 color{1.0F, 0.55F, 0.12F};
    std::int32_t thickness = 2;
};
static_assert(offsetof(OutlinePushConstants, viewport_origin) == 0);
static_assert(offsetof(OutlinePushConstants, viewport_size) == 8);
static_assert(offsetof(OutlinePushConstants, color) == 16);
static_assert(offsetof(OutlinePushConstants, thickness) == 28);
static_assert(sizeof(OutlinePushConstants) <= gpu::GraphicsPipeline::k_push_constant_size);

// Must match the k_operator_* constants in shaders/tonemap.slang.
enum class TonemapOperator : std::int32_t {
    none = 0,
    reinhard = 1,
    aces = 2,
};

// Parallel to the enum above, for the combo box. An array rather than a
// function of the enum so the two orders cannot drift apart unnoticed.
constexpr std::array<const char*, 3> k_tonemap_names{"None (clamp)", "Reinhard", "ACES (fitted)"};

// Must match PushConstants in shaders/tonemap.slang. Verified against the
// compiled SPIR-V, which decorates these members Offset 0 and Offset 4.
struct TonemapPushConstants {
    float exposure = 1.0F;
    std::int32_t tonemap_operator = 0;
};
static_assert(offsetof(TonemapPushConstants, exposure) == 0);
static_assert(offsetof(TonemapPushConstants, tonemap_operator) == 4);
static_assert(sizeof(TonemapPushConstants) <= gpu::GraphicsPipeline::k_push_constant_size);

// Range of the exposure slider, in stops. Four either way covers "the scene is
// lit for a sunny day" to "the scene is lit by one lantern" without the slider
// becoming too coarse to make a small correction with.
constexpr float k_min_exposure_stops = -4.0F;
constexpr float k_max_exposure_stops = 4.0F;

// Captures land here, relative to the working directory. Git-ignored: these are
// output, and a portfolio screenshot belongs in a README by hand rather than
// accumulating in the tree.
constexpr const char* k_screenshot_directory = "screenshots";

// Bytes per pixel in the swapchain format, for sizing the readback buffer.
// Every format is_capturable_format accepts is 8-bit RGBA or BGRA.
constexpr VkDeviceSize k_swapchain_bytes_per_pixel = 4;

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

// A timestamped path under k_screenshot_directory, disambiguated if one already
// exists. The timestamp resolves to the second, so two captures in the same
// second would otherwise silently overwrite each other.
std::filesystem::path next_screenshot_path() {
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm local{};
    localtime_r(&now, &local);

    std::array<char, 32> stamp{};
    std::strftime(stamp.data(), stamp.size(), "%Y%m%d-%H%M%S", &local);

    const std::filesystem::path directory{k_screenshot_directory};
    std::filesystem::path candidate = directory / ("sage-" + std::string(stamp.data()) + ".png");
    for (int suffix = 2; std::filesystem::exists(candidate); ++suffix) {
        candidate = directory /
                    ("sage-" + std::string(stamp.data()) + "-" + std::to_string(suffix) + ".png");
    }
    return candidate;
}

// 64 MiB. The 4 MiB this started at suited one lantern; the picker can now be
// pointed at anything, and additive loads accumulate. Reserved device-local up
// front, which is comfortable on any discrete GPU and fine on integrated.
// Overrunning it drops primitives with a message rather than aborting.
constexpr VkDeviceSize k_geometry_capacity = 64ULL * 1024 * 1024;

// Default dock layout, as fractions of the node being split. The left column
// carries the read-out and the file picker, the right the scene tree, and what
// is left in the middle is the 3D view.
constexpr float k_left_column_fraction = 0.22F;
// Of the remainder after the left column, so ~22% of the window.
constexpr float k_right_column_fraction = 0.28F;
// The stats read-out is a handful of lines, plus the tonemap and capture
// controls; the picker below it wants the rest.
constexpr float k_stats_fraction = 0.5F;
// The right column is split between the scene tree and the properties of
// whatever is selected in it.
constexpr float k_hierarchy_fraction = 0.5F;

// Fraction of the bounding sphere's fitted distance to back off by, so the
// model does not touch the edges of the view.
constexpr float k_framing_margin = 1.15F;

// The dockspace's central node in framebuffer pixels, or the whole image when
// there is no layout yet. Free rather than a member so ImGuiID stays out of
// application.hpp.
VkRect2D central_node_rect(ImGuiID dockspace, VkExtent2D extent) {
    const VkRect2D whole{{0, 0}, extent};

    const ImGuiDockNode* central = ImGui::DockBuilderGetCentralNode(dockspace);
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    if (central == nullptr || central->Size.x <= 0.0F || central->Size.y <= 0.0F ||
        viewport->Size.x <= 0.0F || viewport->Size.y <= 0.0F) {
        return whole;
    }

    // ImGui measures in the window's logical coordinates while the swapchain is
    // in framebuffer pixels. The two are equal until display scaling is
    // involved -- exactly the kind of difference that goes unnoticed on one
    // machine and misplaces the whole viewport on another.
    const float scale_x = static_cast<float>(extent.width) / viewport->Size.x;
    const float scale_y = static_cast<float>(extent.height) / viewport->Size.y;

    // Relative to the viewport rather than absolute: with multi-viewport off
    // the main viewport sits at the origin, but subtracting costs nothing and
    // is correct either way.
    const float x = (central->Pos.x - viewport->Pos.x) * scale_x;
    const float y = (central->Pos.y - viewport->Pos.y) * scale_y;

    // Clamped because a scissor rect reaching outside the attachment is
    // invalid, and rounding at the edges is enough to put it there.
    const auto clamp_to = [](float value, std::uint32_t limit) {
        return static_cast<std::uint32_t>(std::clamp(value, 0.0F, static_cast<float>(limit)));
    };
    const std::uint32_t left = clamp_to(x, extent.width);
    const std::uint32_t top = clamp_to(y, extent.height);

    VkRect2D rect{};
    rect.offset = {static_cast<std::int32_t>(left), static_cast<std::int32_t>(top)};
    rect.extent = {clamp_to(central->Size.x * scale_x, extent.width - left),
                   clamp_to(central->Size.y * scale_y, extent.height - top)};

    // A degenerate node -- collapsed, or mid-resize -- would divide by zero in
    // the projection and make an invalid viewport.
    if (rect.extent.width == 0 || rect.extent.height == 0) {
        return whole;
    }
    return rect;
}

}  // namespace

Application::Application(const std::filesystem::path& model_path)
    // Opens where the model on the command line lives, so the argument still
    // means something; with no argument it starts in the vendored assets, which
    // is the only directory this build knows about.
    : file_picker_(model_path.empty() ? std::filesystem::path(SAGE_ASSET_DIR)
                                      : model_path.parent_path()),
      camera_(k_initial_camera_position, k_initial_camera_yaw, k_initial_camera_pitch),
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
      selection_buffer_(allocator_, uploader_, bindless_set_),
      frame_buffer_(allocator_, device_, sizeof(FrameData) * gpu::FramePacer::k_frames_in_flight,
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT),
      pick_buffer_(allocator_, device_, sizeof(std::uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   gpu::BufferAccess::host_read),
      swapchain_(device_, surface_.handle(), window_.framebuffer_extent()),
      depth_buffer_(allocator_, device_, swapchain_.extent()),
      id_buffer_(allocator_, device_, swapchain_.extent()),
      hdr_target_(allocator_, device_, swapchain_.extent()),
      pipeline_cache_(device_, pipeline_cache_path()),
      pipeline_(device_,
                gpu::GraphicsPipelineDesc{
                    .spirv_path = std::filesystem::path(SAGE_SHADER_DIR) / "mesh.spv",
                    // The HDR target, not the swapchain: this pass no longer
                    // writes anything a display sees directly.
                    .color_format = gpu::HdrTarget::format(),
                    .id_format = gpu::IdBuffer::format(),
                    .depth_format = depth_buffer_.format(),
                    .set_layout = bindless_set_.layout(),
                    .cache = pipeline_cache_.handle(),
                }),
      outline_pipeline_(device_,
                        gpu::GraphicsPipelineDesc{
                            .spirv_path = std::filesystem::path(SAGE_SHADER_DIR) / "outline.spv",
                            // Still the swapchain. The outline is an editor
                            // affordance, not a lit surface: drawing it into
                            // the HDR target would put it through the tonemap,
                            // which would quietly change the colour asked for
                            // into a darker, less saturated one.
                            .color_format = swapchain_.format(),
                            // No id attachment and no depth: this pass reads
                            // ids, it does not write them, and a full-screen
                            // triangle has nothing to be occluded by.
                            .alpha_blend = true,
                            .cull_backfaces = false,
                            .set_layout = bindless_set_.layout(),
                            .cache = pipeline_cache_.handle(),
                        }),
      tonemap_pipeline_(device_,
                        gpu::GraphicsPipelineDesc{
                            .spirv_path = std::filesystem::path(SAGE_SHADER_DIR) / "tonemap.spv",
                            .color_format = swapchain_.format(),
                            .cull_backfaces = false,
                            .set_layout = bindless_set_.layout(),
                            .cache = pipeline_cache_.handle(),
                        }),
      frame_pacer_(device_),
      imgui_(instance_, device_, window_, swapchain_.format(), swapchain_.image_count()) {
    bindless_set_.write_object_id_image(id_buffer_.view());
    bindless_set_.write_hdr_color_image(hdr_target_.view());
    gizmo_operation_ = static_cast<int>(ImGuizmo::TRANSLATE);
    tonemap_operator_ = static_cast<int>(TonemapOperator::aces);

    if (!gpu::is_capturable_format(swapchain_.format())) {
        SAGE_LOG_WARN("Swapchain format {} cannot be captured; screenshots are disabled",
                      static_cast<int>(swapchain_.format()));
    }

    if (model_path.empty()) {
        SAGE_LOG_INFO("No model given; use the Load glTF panel to pick one");
        return;
    }

    // Through the same path a picker click takes, so a command-line model gets
    // no special handling and a bad argument is a message rather than a crash
    // before the window is ever useful.
    if (!load_model(model_path, true)) {
        SAGE_LOG_WARN("Starting with an empty scene");
    }
}

bool Application::load_model(const std::filesystem::path& path, bool replace) {
    if (replace) {
        clear_scene();
    }

    const std::optional<gpu::LoadedScene> loaded = gpu::load_gltf(
        path, geometry_registry_, texture_registry_, material_registry_, scene_graph_);
    if (!loaded.has_value()) {
        return false;
    }

    // Only when the load actually drew something and had the viewport to
    // itself. Re-framing on an additive load would yank the camera away from
    // whatever the user was looking at to fit a model they just added.
    //
    // Queued rather than applied: framing needs the central node's aspect, and
    // the dockspace does not exist yet when the constructor loads a model named
    // on the command line.
    if (replace && loaded->mesh_count > 0) {
        pending_frame_ = SceneBounds{loaded->bounds_min, loaded->bounds_max};
    }
    return true;
}

void Application::clear_scene() {
    // Before anything is torn down: in-flight command buffers still hold this
    // geometry's device addresses, and TextureRegistry::reset destroys images
    // a submitted draw may still sample.
    device_.wait_idle();

    select(gpu::NodeHandle{});
    scene_graph_.clear();
    geometry_registry_.reset();
    material_registry_.reset();
    texture_registry_.reset();
}

void Application::service_pending_load() {
    if (pending_clear_) {
        pending_clear_ = false;
        clear_scene();
        SAGE_LOG_INFO("Scene cleared");
    }

    if (!pending_load_.has_value()) {
        return;
    }

    const FilePicker::Request request = *pending_load_;
    pending_load_.reset();

    if (!load_model(request.path, request.replace)) {
        SAGE_LOG_ERROR("Could not load {}", request.path.string());
    }
}

Application::~Application() {
    // Everything below must outlive in-flight GPU work.
    device_.wait_idle();
}

Application::CameraMatrices Application::camera_matrices() const {
    const float aspect = static_cast<float>(viewport_rect_.extent.width) /
                         static_cast<float>(viewport_rect_.extent.height);
    const float fov = glm::radians(k_field_of_view_degrees);

    CameraMatrices matrices;
    matrices.view = camera_.view_matrix();
    matrices.projection = core::perspective_vk(fov, aspect, k_near_plane, k_far_plane);
    matrices.projection_gl = core::perspective_gl(fov, aspect, k_near_plane, k_far_plane);
    return matrices;
}

void Application::record_scene(VkCommandBuffer command_buffer, std::uint32_t frame_slot) const {
    VkImageMemoryBarrier2 to_color{};
    to_color.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    // FRAGMENT_SHADER as the source stage, not COLOR_ATTACHMENT_OUTPUT: the
    // previous frame's last use of this image was the tonemap sampling it, and
    // that read has to finish before this frame overwrites it.
    to_color.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    to_color.srcAccessMask = VK_ACCESS_2_NONE;
    to_color.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    to_color.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    to_color.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_color.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    to_color.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_color.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_color.image = hdr_target_.image();
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

    // Same shape as the colour barrier: the previous contents are never read,
    // so UNDEFINED discards them and the clear below writes every texel.
    VkImageMemoryBarrier2 to_id{};
    to_id.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_id.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    to_id.srcAccessMask = VK_ACCESS_2_NONE;
    to_id.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    to_id.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    to_id.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_id.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    to_id.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_id.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_id.image = id_buffer_.image();
    to_id.subresourceRange = k_color_range;

    const std::array<VkImageMemoryBarrier2, 3> begin_barriers{to_color, to_depth, to_id};

    VkDependencyInfo begin_dependency{};
    begin_dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    begin_dependency.imageMemoryBarrierCount = static_cast<std::uint32_t>(begin_barriers.size());
    begin_dependency.pImageMemoryBarriers = begin_barriers.data();
    vkCmdPipelineBarrier2(command_buffer, &begin_dependency);

    // loadOp CLEAR is what replaces M1's vkCmdClearColorImage.
    VkRenderingAttachmentInfo color_attachment{};
    color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachment.imageView = hdr_target_.view();
    color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    // The clear covers the whole image while the draws below are scissored to
    // the viewport. That is deliberate: it leaves every texel the tonemap will
    // read defined, so the resolve can be one full-screen triangle with no
    // special case for the region behind the panels.
    color_attachment.clearValue.color = k_clear_color;

    VkRenderingAttachmentInfo depth_attachment{};
    depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_attachment.imageView = depth_buffer_.view();
    depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.clearValue.depthStencil = {1.0F, 0};

    // Cleared to k_null_id, so every pixel no draw covers reads back as
    // "nothing here" rather than as whatever the last frame left behind.
    VkRenderingAttachmentInfo id_attachment{};
    id_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    id_attachment.imageView = id_buffer_.view();
    id_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    id_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    id_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    id_attachment.clearValue.color.uint32[0] = gpu::IdBuffer::k_null_id;

    const std::array<VkRenderingAttachmentInfo, 2> color_attachments{color_attachment,
                                                                     id_attachment};

    VkRenderingInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering.renderArea.offset = {0, 0};
    rendering.renderArea.extent = hdr_target_.extent();
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = static_cast<std::uint32_t>(color_attachments.size());
    rendering.pColorAttachments = color_attachments.data();
    rendering.pDepthAttachment = &depth_attachment;

    vkCmdBeginRendering(command_buffer, &rendering);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.handle());

    const VkDescriptorSet descriptor_set = bindless_set_.handle();
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.layout(), 0,
                            1, &descriptor_set, 0, nullptr);

    // The dockspace's central node, not the whole image: the panels are opaque,
    // so drawing behind them costs fill rate and, worse, makes the projection
    // describe a view wider than the one actually on screen.
    VkViewport viewport{};
    viewport.x = static_cast<float>(viewport_rect_.offset.x);
    viewport.y = static_cast<float>(viewport_rect_.offset.y);
    viewport.width = static_cast<float>(viewport_rect_.extent.width);
    viewport.height = static_cast<float>(viewport_rect_.extent.height);
    viewport.minDepth = 0.0F;
    viewport.maxDepth = 1.0F;
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);

    vkCmdSetScissor(command_buffer, 0, 1, &viewport_rect_);

    const CameraMatrices matrices = camera_matrices();

    // One write per frame, into this frame's own slot. Writing a single shared
    // slot would race the GPU, which may still be reading the previous frame's
    // copy. begin_frame() has already waited out the work that used this slot.
    FrameData frame_data;
    frame_data.view_projection = matrices.projection * matrices.view;
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

    for (std::uint32_t index = 0; index < scene_graph_.nodes().size(); ++index) {
        const gpu::SceneNode& node = scene_graph_.nodes()[index];
        if (!node.has_mesh) {
            // Pure transform nodes: glTF hierarchy nodes, and the per-load root.
            continue;
        }

        PushConstants push{};
        push.vertex_address = node.mesh.vertex_address;
        push.frame_address = frame_address;
        push.model = node.world_transform;
        push.material_index = node.material_index;
        // Offset by one so that k_null_id (0) stays reserved for empty space;
        // resolve_pick subtracts it back off.
        push.object_id = index + 1;
        vkCmdPushConstants(command_buffer, pipeline_.layout(), VK_SHADER_STAGE_ALL, 0, sizeof(push),
                           &push);
        vkCmdBindIndexBuffer(command_buffer, geometry_registry_.buffer(), node.mesh.index_offset,
                             VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(command_buffer, node.mesh.index_count, 1, 0, 0, 0);
    }

    vkCmdEndRendering(command_buffer);
}

void Application::record_tonemap(VkCommandBuffer command_buffer, VkImage image,
                                 VkImageView image_view) const {
    // Two transitions, one dependency. The HDR target stops being an attachment
    // and becomes a texture; the swapchain image, untouched so far this frame,
    // becomes an attachment for the first time.
    VkImageMemoryBarrier2 hdr_to_read{};
    hdr_to_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    hdr_to_read.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    hdr_to_read.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    hdr_to_read.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    hdr_to_read.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    hdr_to_read.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    hdr_to_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    hdr_to_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hdr_to_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hdr_to_read.image = hdr_target_.image();
    hdr_to_read.subresourceRange = k_color_range;

    VkImageMemoryBarrier2 swapchain_to_color{};
    swapchain_to_color.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    swapchain_to_color.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    swapchain_to_color.srcAccessMask = VK_ACCESS_2_NONE;
    swapchain_to_color.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    swapchain_to_color.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    swapchain_to_color.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    swapchain_to_color.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    swapchain_to_color.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapchain_to_color.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapchain_to_color.image = image;
    swapchain_to_color.subresourceRange = k_color_range;

    const std::array<VkImageMemoryBarrier2, 2> barriers{hdr_to_read, swapchain_to_color};

    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.imageMemoryBarrierCount = static_cast<std::uint32_t>(barriers.size());
    dependency.pImageMemoryBarriers = barriers.data();
    vkCmdPipelineBarrier2(command_buffer, &dependency);

    // DONT_CARE, not CLEAR: the triangle below covers every pixel of the image,
    // so clearing first would be writing the whole swapchain twice.
    VkRenderingAttachmentInfo color_attachment{};
    color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachment.imageView = image_view;
    color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    const VkExtent2D extent = swapchain_.extent();

    VkRenderingInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering.renderArea.offset = {0, 0};
    rendering.renderArea.extent = extent;
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &color_attachment;

    vkCmdBeginRendering(command_buffer, &rendering);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemap_pipeline_.handle());

    const VkDescriptorSet descriptor_set = bindless_set_.handle();
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            tonemap_pipeline_.layout(), 0, 1, &descriptor_set, 0, nullptr);

    // The whole image, not viewport_rect_: the region behind the panels was
    // cleared by the scene pass and still has to be resolved, or it would show
    // whatever the presentation engine last left in this swapchain image.
    VkViewport viewport{};
    viewport.x = 0.0F;
    viewport.y = 0.0F;
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0F;
    viewport.maxDepth = 1.0F;
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);

    const VkRect2D scissor{{0, 0}, extent};
    vkCmdSetScissor(command_buffer, 0, 1, &scissor);

    TonemapPushConstants push{};
    // Stops are a doubling each, which is what the exp2 is: +1 stop is twice
    // the light reaching the sensor.
    push.exposure = std::exp2(exposure_stops_);
    push.tonemap_operator = tonemap_operator_;
    vkCmdPushConstants(command_buffer, tonemap_pipeline_.layout(), VK_SHADER_STAGE_ALL, 0,
                       sizeof(push), &push);

    vkCmdDraw(command_buffer, 3, 1, 0, 0);
    vkCmdEndRendering(command_buffer);
}

void Application::handle_picking_input() {
    // A panel under the pointer wins, matching the camera's rule -- otherwise
    // clicking a tree row would also pick whatever is behind the panel.
    if (gpu::ImGuiLayer::wants_mouse() || !ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        return;
    }
    // Right button held means the camera is being flown, and a left click then
    // is part of that gesture rather than a selection.
    if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        return;
    }

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    if (viewport->Size.x <= 0.0F || viewport->Size.y <= 0.0F) {
        return;
    }

    // ImGui reports logical coordinates; the id attachment is framebuffer
    // pixels. Same conversion as central_node_rect, for the same reason.
    const VkExtent2D extent = id_buffer_.extent();
    const ImVec2 mouse = ImGui::GetMousePos();
    const float x =
        (mouse.x - viewport->Pos.x) * (static_cast<float>(extent.width) / viewport->Size.x);
    const float y =
        (mouse.y - viewport->Pos.y) * (static_cast<float>(extent.height) / viewport->Size.y);

    const auto texel_x = static_cast<std::int32_t>(x);
    const auto texel_y = static_cast<std::int32_t>(y);

    // Outside the 3D view is not a miss, it is not a pick at all -- clicking
    // the dockspace border should leave the selection alone.
    const VkRect2D& rect = viewport_rect_;
    if (texel_x < rect.offset.x || texel_y < rect.offset.y ||
        texel_x >= rect.offset.x + static_cast<std::int32_t>(rect.extent.width) ||
        texel_y >= rect.offset.y + static_cast<std::int32_t>(rect.extent.height)) {
        return;
    }

    pending_pick_ = VkOffset2D{texel_x, texel_y};
}

void Application::handle_gizmo_keys() {
    // Gated on the camera not being flown: W/E/R are also part of WASD, and
    // holding the right button means the keys belong to the camera.
    if (gpu::ImGuiLayer::wants_keyboard() || ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        return;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_W)) {
        gizmo_operation_ = static_cast<int>(ImGuizmo::TRANSLATE);
    } else if (ImGui::IsKeyPressed(ImGuiKey_E)) {
        gizmo_operation_ = static_cast<int>(ImGuizmo::ROTATE);
    } else if (ImGui::IsKeyPressed(ImGuiKey_R)) {
        gizmo_operation_ = static_cast<int>(ImGuizmo::SCALE);
    }
}

void Application::select(gpu::NodeHandle node) {
    if (node == selected_) {
        return;
    }
    selected_ = node;
    selection_dirty_ = true;
}

void Application::service_selection() {
    if (!selection_dirty_) {
        return;
    }
    selection_dirty_ = false;

    scene_graph_.mark_subtree(selected_, selection_flags_);
    selection_buffer_.update(selection_flags_);
}

void Application::record_outline(VkCommandBuffer command_buffer, VkImageView image_view) const {
    // Unconditional, even with nothing selected: the barrier below is what
    // leaves the id image in the layout record_pick_copy expects, so skipping
    // the pass would make that layout depend on the selection.
    VkImageMemoryBarrier2 to_read{};
    to_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_read.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    to_read.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    to_read.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    to_read.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    to_read.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    to_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    to_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_read.image = id_buffer_.image();
    to_read.subresourceRange = k_color_range;

    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &to_read;
    vkCmdPipelineBarrier2(command_buffer, &dependency);

    if (!selected_.valid()) {
        return;
    }

    // LOAD, not CLEAR: the shaded scene is already here and the outline is
    // drawn over it.
    VkRenderingAttachmentInfo color_attachment{};
    color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachment.imageView = image_view;
    color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering.renderArea.offset = viewport_rect_.offset;
    rendering.renderArea.extent = viewport_rect_.extent;
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &color_attachment;

    vkCmdBeginRendering(command_buffer, &rendering);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, outline_pipeline_.handle());

    const VkDescriptorSet descriptor_set = bindless_set_.handle();
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            outline_pipeline_.layout(), 0, 1, &descriptor_set, 0, nullptr);

    VkViewport viewport{};
    viewport.x = static_cast<float>(viewport_rect_.offset.x);
    viewport.y = static_cast<float>(viewport_rect_.offset.y);
    viewport.width = static_cast<float>(viewport_rect_.extent.width);
    viewport.height = static_cast<float>(viewport_rect_.extent.height);
    viewport.minDepth = 0.0F;
    viewport.maxDepth = 1.0F;
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(command_buffer, 0, 1, &viewport_rect_);

    OutlinePushConstants push{};
    push.viewport_origin = {viewport_rect_.offset.x, viewport_rect_.offset.y};
    push.viewport_size = {static_cast<std::int32_t>(viewport_rect_.extent.width),
                          static_cast<std::int32_t>(viewport_rect_.extent.height)};
    vkCmdPushConstants(command_buffer, outline_pipeline_.layout(), VK_SHADER_STAGE_ALL, 0,
                       sizeof(push), &push);

    // Three vertices, no buffers: the vertex shader builds a full-screen
    // triangle from SV_VertexID.
    vkCmdDraw(command_buffer, 3, 1, 0, 0);
    vkCmdEndRendering(command_buffer);
}

void Application::record_pick_copy(VkCommandBuffer command_buffer) const {
    if (!pending_pick_.has_value()) {
        return;
    }

    VkImageMemoryBarrier2 to_transfer_src{};
    to_transfer_src.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    // SHADER_READ_ONLY, not COLOR_ATTACHMENT: record_outline has already moved
    // the image there. Ordering the two passes rather than making each one
    // handle both cases keeps a single source layout per barrier.
    to_transfer_src.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    to_transfer_src.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    to_transfer_src.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    to_transfer_src.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    to_transfer_src.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    to_transfer_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_transfer_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_transfer_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_transfer_src.image = id_buffer_.image();
    to_transfer_src.subresourceRange = k_color_range;

    VkDependencyInfo to_transfer_dependency{};
    to_transfer_dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    to_transfer_dependency.imageMemoryBarrierCount = 1;
    to_transfer_dependency.pImageMemoryBarriers = &to_transfer_src;
    vkCmdPipelineBarrier2(command_buffer, &to_transfer_dependency);

    // One texel. The whole point of an id attachment over a CPU-side raycast is
    // that the answer is already rendered; reading more of it would be waste.
    VkBufferImageCopy2 region{};
    region.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {pending_pick_->x, pending_pick_->y, 0};
    region.imageExtent = {1, 1, 1};

    VkCopyImageToBufferInfo2 copy{};
    copy.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2;
    copy.srcImage = id_buffer_.image();
    copy.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    copy.dstBuffer = pick_buffer_.handle();
    copy.regionCount = 1;
    copy.pRegions = &region;
    vkCmdCopyImageToBuffer2(command_buffer, &copy);

    // Waiting on the submission alone does not make the copy visible to the
    // host; the HOST stage has to be named as a destination for that.
    VkBufferMemoryBarrier2 to_host{};
    to_host.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    to_host.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    to_host.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    to_host.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    to_host.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    to_host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_host.buffer = pick_buffer_.handle();
    to_host.offset = 0;
    to_host.size = VK_WHOLE_SIZE;

    VkDependencyInfo to_host_dependency{};
    to_host_dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    to_host_dependency.bufferMemoryBarrierCount = 1;
    to_host_dependency.pBufferMemoryBarriers = &to_host;
    vkCmdPipelineBarrier2(command_buffer, &to_host_dependency);
}

void Application::resolve_pick() {
    if (!pending_pick_.has_value()) {
        return;
    }
    pending_pick_.reset();

    // The copy is in a submission that has to have completed before the mapping
    // holds this frame's value. Blocking is deliberate: it costs a stall on the
    // frame a click happens, where the alternative -- reading a slot two frames
    // old -- would hand back whatever was under the cursor before the click.
    frame_pacer_.wait_all();

    std::uint32_t id = gpu::IdBuffer::k_null_id;
    pick_buffer_.read(&id, sizeof(id));

    // Promoted to the root of whatever was hit. Clicking a mesh selects the
    // object it belongs to, not the individual submesh -- which is what makes
    // a loaded file behave as one thing to move, and what the properties panel
    // will want to describe.
    const gpu::NodeHandle hit =
        id == gpu::IdBuffer::k_null_id ? gpu::NodeHandle{} : scene_graph_.handle_at(id - 1);
    select(scene_graph_.root_of(hit));

    if (const gpu::SceneNode* node = scene_graph_.find(selected_); node != nullptr) {
        SAGE_LOG_INFO("Selected '{}'", node->name);
    } else {
        SAGE_LOG_INFO("Selection cleared");
    }
}

void Application::request_screenshot() {
    if (pending_screenshot_.has_value()) {
        // One in flight already. Queuing a second would overwrite the readback
        // buffer the first is still waiting on.
        return;
    }
    if (!gpu::is_capturable_format(swapchain_.format())) {
        SAGE_LOG_WARN("Swapchain format cannot be captured");
        return;
    }
    pending_screenshot_ = next_screenshot_path();
}

void Application::record_screenshot_copy(VkCommandBuffer command_buffer, VkImage image) {
    if (!pending_screenshot_.has_value()) {
        return;
    }

    // The 3D view alone. Everything outside it is either the background the
    // scene pass cleared or, once ImGui has drawn, editor chrome -- and this
    // runs before ImGui anyway, so the panels would be blank rectangles.
    const VkRect2D region_rect = viewport_rect_;
    screenshot_extent_ = region_rect.extent;

    const VkDeviceSize size = VkDeviceSize{screenshot_extent_.width} * screenshot_extent_.height *
                              k_swapchain_bytes_per_pixel;
    screenshot_buffer_ = std::make_unique<gpu::Buffer>(
        allocator_, device_, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpu::BufferAccess::host_read);

    VkImageMemoryBarrier2 to_transfer_src{};
    to_transfer_src.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_transfer_src.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    to_transfer_src.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    to_transfer_src.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    to_transfer_src.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    to_transfer_src.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    to_transfer_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_transfer_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_transfer_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_transfer_src.image = image;
    to_transfer_src.subresourceRange = k_color_range;

    VkDependencyInfo to_transfer_dependency{};
    to_transfer_dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    to_transfer_dependency.imageMemoryBarrierCount = 1;
    to_transfer_dependency.pImageMemoryBarriers = &to_transfer_src;
    vkCmdPipelineBarrier2(command_buffer, &to_transfer_dependency);

    // bufferRowLength 0 means "rows are packed to imageExtent", which is the
    // layout write_png expects.
    VkBufferImageCopy2 region{};
    region.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {region_rect.offset.x, region_rect.offset.y, 0};
    region.imageExtent = {screenshot_extent_.width, screenshot_extent_.height, 1};

    VkCopyImageToBufferInfo2 copy{};
    copy.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2;
    copy.srcImage = image;
    copy.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    copy.dstBuffer = screenshot_buffer_->handle();
    copy.regionCount = 1;
    copy.pRegions = &region;
    vkCmdCopyImageToBuffer2(command_buffer, &copy);

    // Back to an attachment: the outline and then ImGui still draw into this
    // image, and transition_to_present expects to find it in that layout.
    VkImageMemoryBarrier2 to_color{};
    to_color.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_color.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    to_color.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    to_color.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    // READ as well as WRITE: the outline pass that follows begins with
    // loadOp LOAD, which reads the attachment before blending over it. Granting
    // only write access here is a read-after-write hazard that ordinary
    // validation does not see and synchronisation validation does.
    to_color.dstAccessMask =
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
    to_color.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_color.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    to_color.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_color.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_color.image = image;
    to_color.subresourceRange = k_color_range;

    // Same reasoning as the pick copy: waiting on the submission alone does not
    // make the copy visible to the host. The HOST stage has to be named.
    VkBufferMemoryBarrier2 to_host{};
    to_host.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    to_host.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    to_host.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    to_host.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    to_host.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    to_host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_host.buffer = screenshot_buffer_->handle();
    to_host.offset = 0;
    to_host.size = VK_WHOLE_SIZE;

    VkDependencyInfo back_dependency{};
    back_dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    back_dependency.bufferMemoryBarrierCount = 1;
    back_dependency.pBufferMemoryBarriers = &to_host;
    back_dependency.imageMemoryBarrierCount = 1;
    back_dependency.pImageMemoryBarriers = &to_color;
    vkCmdPipelineBarrier2(command_buffer, &back_dependency);
}

void Application::resolve_screenshot() {
    if (!pending_screenshot_.has_value() || screenshot_buffer_ == nullptr) {
        return;
    }

    // Blocking, like resolve_pick: a capture is a deliberate one-off, so the
    // simpler code is worth more than the frame it costs.
    frame_pacer_.wait_all();

    std::vector<std::byte> pixels(screenshot_buffer_->size());
    screenshot_buffer_->read(pixels.data(), screenshot_buffer_->size());

    static_cast<void>(
        gpu::write_png(*pending_screenshot_, screenshot_extent_, swapchain_.format(), pixels));

    pending_screenshot_.reset();
    screenshot_buffer_.reset();
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

void Application::draw_dockspace() {
    // PassthruCentralNode leaves the middle node transparent and, while it is
    // empty, lets mouse input through it -- so the scene shows and the camera
    // still responds. NoDockingOverCentralNode keeps it empty permanently, so a
    // panel cannot be dragged over the 3D view by accident.
    const ImGuiID dockspace = ImGui::DockSpaceOverViewport(
        0, ImGui::GetMainViewport(),
        ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_NoDockingOverCentralNode);

    viewport_rect_ = central_node_rect(dockspace, swapchain_.extent());

    if (dock_layout_built_) {
        return;
    }
    dock_layout_built_ = true;

    // Rebuilt from scratch every run. Layout persistence would be io.IniFilename,
    // which ImGuiLayer deliberately leaves null, so there is nothing to preserve
    // and the arrangement is the same on every start.
    ImGui::DockBuilderRemoveNode(dockspace);
    ImGui::DockBuilderAddNode(dockspace, ImGuiDockNodeFlags_DockSpace);
    // Set before splitting: the ratios below are fractions of the node's size,
    // and a node that has not been sized yet splits unpredictably.
    ImGui::DockBuilderSetNodeSize(dockspace, ImGui::GetMainViewport()->Size);

    ImGuiID left = 0;
    ImGuiID centre = 0;
    ImGui::DockBuilderSplitNode(dockspace, ImGuiDir_Left, k_left_column_fraction, &left, &centre);

    ImGuiID right = 0;
    ImGui::DockBuilderSplitNode(centre, ImGuiDir_Right, k_right_column_fraction, &right, &centre);

    ImGuiID left_top = 0;
    ImGuiID left_bottom = 0;
    ImGui::DockBuilderSplitNode(left, ImGuiDir_Up, k_stats_fraction, &left_top, &left_bottom);

    ImGui::DockBuilderDockWindow("sage", left_top);
    ImGui::DockBuilderDockWindow("Load glTF", left_bottom);
    ImGuiID right_top = 0;
    ImGuiID right_bottom = 0;
    ImGui::DockBuilderSplitNode(right, ImGuiDir_Up, k_hierarchy_fraction, &right_top,
                                &right_bottom);

    ImGui::DockBuilderDockWindow("Hierarchy", right_top);
    ImGui::DockBuilderDockWindow("Properties", right_bottom);
    ImGui::DockBuilderFinish(dockspace);

    // The split above changed the central node, so the rect taken before it is
    // stale for this frame.
    viewport_rect_ = central_node_rect(dockspace, swapchain_.extent());
}

void Application::draw_ui() {
    const ImGuiIO& io = ImGui::GetIO();

    ImGui::Begin("sage");
    ImGui::Text("%.1f fps (%.2f ms)", static_cast<double>(io.Framerate),
                1000.0 / static_cast<double>(io.Framerate));
    ImGui::Separator();
    ImGui::Text("Scene: %zu nodes", scene_graph_.size());
    ImGui::Text("Geometry: %llu / %llu KiB",
                static_cast<unsigned long long>(geometry_registry_.used() / 1024),
                static_cast<unsigned long long>(geometry_registry_.capacity() / 1024));
    ImGui::Text("Materials: %u / %u", material_registry_.count(), material_registry_.capacity());
    ImGui::Text("Textures: %u", texture_registry_.count());
    const gpu::SceneNode* selected = scene_graph_.find(selected_);
    ImGui::Text("Selected: %s", selected != nullptr ? selected->name.c_str() : "(none)");
    const glm::vec3 position = camera_.position();
    ImGui::Text("Camera: %.1f, %.1f, %.1f", static_cast<double>(position.x),
                static_cast<double>(position.y), static_cast<double>(position.z));
    draw_presentation_controls();
    ImGui::End();

    // Queued rather than serviced here: this is the middle of a frame, and the
    // load blocks, waits for the device and destroys images the command buffer
    // being recorded would still reference.
    if (std::optional<FilePicker::Request> request = file_picker_.draw(); request.has_value()) {
        pending_load_ = std::move(request);
    }
    if (file_picker_.clear_requested()) {
        // Same reasoning: clear_scene waits for the device to be idle, which is
        // not something to do with a frame half-recorded.
        pending_clear_ = true;
    }
}

void Application::draw_presentation_controls() {
    ImGui::SeparatorText("Tonemap");

    // Selectable at runtime rather than baked in, because the difference between
    // two curves is only legible on the same frame -- comparing across a rebuild
    // compares two memories of an image.
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::Combo("##tonemap", &tonemap_operator_, k_tonemap_names.data(),
                 static_cast<int>(k_tonemap_names.size()));
    if (tonemap_operator_ == static_cast<int>(TonemapOperator::none)) {
        ImGui::TextDisabled("Clamped, as before M8");
    }

    ImGui::SliderFloat("Exposure", &exposure_stops_, k_min_exposure_stops, k_max_exposure_stops,
                       "%+.2f stops");

    ImGui::SeparatorText("Capture");
    // Disabled rather than hidden while one is in flight: the button vanishing
    // for a frame reads as a misclick.
    ImGui::BeginDisabled(pending_screenshot_.has_value());
    if (ImGui::Button("Screenshot (F2)", ImVec2(-FLT_MIN, 0.0F))) {
        request_screenshot();
    }
    ImGui::EndDisabled();
    ImGui::TextDisabled("%ux%u, no UI", viewport_rect_.extent.width, viewport_rect_.extent.height);
}

void Application::draw_properties_panel() {
    ImGui::Begin("Properties");

    const gpu::SceneNode* node = scene_graph_.find(selected_);
    if (node == nullptr) {
        ImGui::TextDisabled("Nothing selected.");
        ImGui::TextDisabled("Click an object, or a row in the Hierarchy.");
        ImGui::End();
        return;
    }

    ImGui::Text("%s", node->name.c_str());
    ImGui::Separator();

    // Decomposed with ImGuizmo's own helper rather than glm's, so the numbers
    // shown here and the numbers a drag produces come from the same code. Two
    // decompositions that disagree on, say, euler order would make the panel
    // jitter while dragging.
    glm::mat4 local = node->local_transform;
    glm::vec3 translation{0.0F};
    glm::vec3 rotation{0.0F};
    glm::vec3 scale{1.0F};
    ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(local), glm::value_ptr(translation),
                                          glm::value_ptr(rotation), glm::value_ptr(scale));

    bool edited = false;
    edited |= ImGui::DragFloat3("Position", glm::value_ptr(translation), 0.01F);
    edited |= ImGui::DragFloat3("Rotation", glm::value_ptr(rotation), 0.5F);
    edited |= ImGui::DragFloat3("Scale", glm::value_ptr(scale), 0.01F);

    if (edited) {
        // A zero on any axis makes the matrix singular, which the next
        // decomposition cannot undo -- the node would be stuck flat.
        scale = glm::max(scale, glm::vec3(1e-4F));
        ImGuizmo::RecomposeMatrixFromComponents(glm::value_ptr(translation),
                                                glm::value_ptr(rotation), glm::value_ptr(scale),
                                                glm::value_ptr(local));
        scene_graph_.set_local_transform(selected_, local);
        scene_graph_.update_transforms();
    }

    ImGui::Separator();
    ImGui::Text("Mesh: %s", node->has_mesh ? "yes" : "no");
    if (node->has_mesh) {
        ImGui::Text("Indices: %u", node->mesh.index_count);
        ImGui::Text("Material: %u", node->material_index);
    }

    ImGui::Separator();
    // Radio buttons rather than a combo: three options that change with one
    // click, and the keyboard shortcuts below mirror them.
    int operation = gizmo_operation_;
    ImGui::TextUnformatted("Gizmo");
    ImGui::RadioButton("Move (W)", &operation, static_cast<int>(ImGuizmo::TRANSLATE));
    ImGui::SameLine();
    ImGui::RadioButton("Rotate (E)", &operation, static_cast<int>(ImGuizmo::ROTATE));
    ImGui::SameLine();
    ImGui::RadioButton("Scale (R)", &operation, static_cast<int>(ImGuizmo::SCALE));
    gizmo_operation_ = operation;

    // Scale is always along the object's own axes; offering a world-space
    // scale would just be a lie about what the gizmo does.
    if (gizmo_operation_ != static_cast<int>(ImGuizmo::SCALE)) {
        ImGui::Checkbox("Local space", &gizmo_local_space_);
    }

    ImGui::End();
}

bool Application::draw_gizmo() {
    const gpu::SceneNode* node = scene_graph_.find(selected_);
    if (node == nullptr) {
        return false;
    }

    ImGuizmo::SetDrawlist(ImGui::GetBackgroundDrawList());
    ImGuizmo::SetOrthographic(false);

    // ImGui's logical coordinates, not framebuffer pixels: the gizmo is drawn
    // through ImGui's draw list, which works in the same space the mouse does.
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const VkExtent2D extent = swapchain_.extent();
    const float to_logical_x = viewport->Size.x / static_cast<float>(extent.width);
    const float to_logical_y = viewport->Size.y / static_cast<float>(extent.height);
    ImGuizmo::SetRect(viewport->Pos.x + static_cast<float>(viewport_rect_.offset.x) * to_logical_x,
                      viewport->Pos.y + static_cast<float>(viewport_rect_.offset.y) * to_logical_y,
                      static_cast<float>(viewport_rect_.extent.width) * to_logical_x,
                      static_cast<float>(viewport_rect_.extent.height) * to_logical_y);

    const CameraMatrices matrices = camera_matrices();

    // The gizmo manipulates a world transform, which is what makes dragging a
    // child behave the way the screen suggests rather than in its parent's
    // rotated frame.
    glm::mat4 world = node->world_transform;

    const bool changed = ImGuizmo::Manipulate(
        glm::value_ptr(matrices.view), glm::value_ptr(matrices.projection_gl),
        static_cast<ImGuizmo::OPERATION>(gizmo_operation_),
        gizmo_local_space_ ? ImGuizmo::LOCAL : ImGuizmo::WORLD, glm::value_ptr(world));

    if (changed) {
        // Back out of world space into the parent's. The parent's world
        // transform is already final this frame -- parents precede children --
        // so no re-composition is needed before inverting it.
        glm::mat4 parent_world{1.0F};
        if (const gpu::SceneNode* parent = scene_graph_.find(node->parent); parent != nullptr) {
            parent_world = parent->world_transform;
        }
        scene_graph_.set_local_transform(selected_, glm::inverse(parent_world) * world);
        scene_graph_.update_transforms();
    }

    return ImGuizmo::IsUsing();
}

void Application::draw_hierarchy_panel() {
    ImGui::Begin("Hierarchy");

    const std::span<const gpu::SceneNode> nodes = scene_graph_.nodes();

    if (nodes.empty()) {
        ImGui::TextDisabled("Empty scene -- load a glTF file.");
        ImGui::End();
        return;
    }

    // SceneNode stores only its parent, and a tree widget needs the inverse.
    // Rebuilt per frame rather than kept in the graph: a second structure there
    // would need maintaining on every add and every removal, and would exist
    // for this panel alone. One linear pass over a few hundred nodes is free.
    ChildTable children(nodes.size());
    std::vector<std::uint32_t> roots;
    for (std::uint32_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].parent.valid()) {
            children[nodes[i].parent.index()].push_back(i);
        } else {
            roots.push_back(i);
        }
    }

    for (const std::uint32_t root : roots) {
        draw_hierarchy_node(root, children);
    }

    ImGui::End();
}

void Application::draw_hierarchy_node(std::uint32_t index, const ChildTable& children) {
    const gpu::SceneNode& node = scene_graph_.nodes()[index];

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth |
                               ImGuiTreeNodeFlags_DefaultOpen;
    // Compared by handle rather than index: a stale handle from before a scene
    // reload must not light up whatever now occupies that slot.
    if (selected_.valid() && scene_graph_.handle_at(index) == selected_) {
        flags |= ImGuiTreeNodeFlags_Selected;
    }
    if (children[index].empty()) {
        // A leaf gets no arrow, and NoTreePushOnOpen means it must not be
        // popped -- which is why the TreePop below is guarded on having
        // children rather than on `open` alone. Popping an unpushed tree node
        // corrupts ImGui's id stack and asserts somewhere unrelated later.
        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    }

    // PushID scopes the label-derived id, so two nodes sharing a name stay
    // distinct without having to build unique label strings.
    ImGui::PushID(static_cast<int>(index));
    const bool open = ImGui::TreeNodeEx(node.name.c_str(), flags);

    // Not promoted to the root, unlike a viewport click: the panel exists to
    // reach a specific node, so clicking a child selects that child and
    // outlines its own subtree.
    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
        select(scene_graph_.handle_at(index));
    }

    if (node.has_mesh) {
        ImGui::SameLine();
        ImGui::TextDisabled("(mesh)");
    }

    if (open && !children[index].empty()) {
        for (const std::uint32_t child : children[index]) {
            draw_hierarchy_node(child, children);
        }
        ImGui::TreePop();
    }
    ImGui::PopID();
}

bool Application::recreate_swapchain() {
    VkExtent2D extent = window_.framebuffer_extent();
    if (extent.width == 0 || extent.height == 0) {
        return false;
    }
    swapchain_.recreate(extent);
    depth_buffer_.recreate(swapchain_.extent());
    id_buffer_.recreate(swapchain_.extent());
    hdr_target_.recreate(swapchain_.extent());
    // The old views are gone, so the descriptors naming them have to be
    // rewritten.
    bindless_set_.write_object_id_image(id_buffer_.view());
    bindless_set_.write_hdr_color_image(hdr_target_.view());
    // A pick names a texel in an image that no longer exists.
    pending_pick_.reset();
    // Same for a capture: the region it named was measured against the old
    // viewport, and the buffer it would read was sized for that.
    pending_screenshot_.reset();
    screenshot_buffer_.reset();
    return true;
}

void Application::frame_camera_on(const glm::vec3& bounds_min, const glm::vec3& bounds_max) {
    const glm::vec3 center = (bounds_min + bounds_max) * 0.5F;
    const float radius = glm::length(bounds_max - bounds_min) * 0.5F;

    const float half_fov_y = glm::radians(k_field_of_view_degrees) * 0.5F;
    // The horizontal field of view is the vertical one widened by the aspect
    // ratio. Fitting to the vertical alone was right while the scene filled a
    // 16:9 window; the docked viewport is nearly square, which makes the
    // horizontal the tighter of the two and the one that crops the model.
    const float aspect = static_cast<float>(viewport_rect_.extent.width) /
                         static_cast<float>(viewport_rect_.extent.height);
    const float half_fov_x = std::atan(std::tan(half_fov_y) * aspect);
    const float half_fov = std::min(half_fov_x, half_fov_y);

    // Distance at which the bounding sphere is tangent to the view frustum.
    // Dividing by tan instead fits only the sphere's equatorial disc, which
    // leaves the near cap outside the frustum -- the old 1.5 fudge factor was
    // covering for that.
    const float distance = (radius / std::sin(half_fov)) * k_framing_margin;

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

        // Before anything else this iteration: no command buffer has been begun
        // and no ImGui frame is open, so a wait_idle and a blocking upload here
        // disturb nothing. Costs the picker one frame of latency.
        service_pending_load();
        service_selection();

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
        // After ImGui::NewFrame and before anything queries the gizmo: it
        // caches per-frame state of its own.
        ImGuizmo::BeginFrame();
        // First, so the panels below have a dockspace to place themselves in.
        draw_dockspace();

        // Immediately after, because draw_dockspace is what establishes the
        // viewport rect the framing is computed against.
        if (pending_frame_.has_value()) {
            frame_camera_on(pending_frame_->min, pending_frame_->max);
            pending_frame_.reset();
        }

        handle_gizmo_keys();

        // Not in handle_gizmo_keys: that is gated on the camera not being flown,
        // and there is no reason a capture should be.
        if (!gpu::ImGuiLayer::wants_keyboard() && ImGui::IsKeyPressed(ImGuiKey_F2)) {
            request_screenshot();
        }

        // Before picking: a drag that ends over a different object must not
        // also reselect it, and ImGuizmo only reports IsUsing() once drawn.
        const bool gizmo_active = draw_gizmo();

        // After draw_dockspace, which refreshes viewport_rect_, and before the
        // panels, so a click is tested against this frame's layout.
        if (!gizmo_active) {
            handle_picking_input();
        }

        draw_ui();
        draw_hierarchy_panel();
        draw_properties_panel();

        // Pass order is load-bearing. The scene shades into the HDR target; the
        // tonemap resolves it to the swapchain and is the first thing to touch
        // that image; the capture takes the resolved scene before any editor
        // overlay reaches it; the outline and then ImGui draw on top.
        record_scene(frame.command_buffer, frame.slot);
        record_tonemap(frame.command_buffer, acquired.image, swapchain_.image_view(acquired.index));
        record_screenshot_copy(frame.command_buffer, acquired.image);
        record_outline(frame.command_buffer, swapchain_.image_view(acquired.index));
        record_pick_copy(frame.command_buffer);
        imgui_.render(frame.command_buffer, swapchain_.image_view(acquired.index),
                      swapchain_.extent());
        transition_to_present(frame.command_buffer, acquired.image);

        frame_pacer_.submit(device_.graphics_queue(), frame,
                            swapchain_.render_finished(acquired.index));

        // After submit, so the copies have been handed to the GPU; both of these
        // wait for it before touching their mappings.
        resolve_pick();
        resolve_screenshot();

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
