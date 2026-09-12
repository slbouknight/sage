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
#include <numbers>
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

// Must match FrameData in shaders/mesh.slang *and* shaders/shadow.slang, which
// declare it separately because slangc compiles each file alone. Written once
// per frame and read by every draw, which is exactly why it is a buffer and not
// a push constant.
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
    alignas(16) glm::mat4 light_view_projection{1.0F};
    float shadow_normal_bias = 0.0F;
    std::int32_t shadow_pcf_radius = 0;
    float shadow_texel_size = 0.0F;
    std::uint32_t shadow_enabled = 0;
    float ambient_intensity = 0.0F;
};
static_assert(offsetof(FrameData, view_projection) == 0);
static_assert(offsetof(FrameData, camera_position) == 64);
static_assert(offsetof(FrameData, light_count) == 76);
static_assert(offsetof(FrameData, lights) == 80);
// Verified against the compiled SPIR-V, which decorates these members at
// exactly these offsets in both mesh.slang and shadow.slang.
static_assert(offsetof(FrameData, light_view_projection) == 464);
static_assert(offsetof(FrameData, shadow_normal_bias) == 528);
static_assert(offsetof(FrameData, shadow_pcf_radius) == 532);
static_assert(offsetof(FrameData, shadow_texel_size) == 536);
static_assert(offsetof(FrameData, shadow_enabled) == 540);
static_assert(offsetof(FrameData, ambient_intensity) == 544);
// 560, not 548: the alignas(16) on the two matrices gives the whole struct
// 16-byte alignment, so its size rounds up. The trailing 12 bytes are padding
// no shader reads.
static_assert(sizeof(FrameData) == 560);
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

// Must match PushConstants in shaders/fxaa.slang. Verified against the compiled
// SPIR-V, which decorates these members at 0, 8, 12, 16 and 20 -- the float2
// aligns to 8, which is what puts the first scalar at 8 rather than 4.
struct FxaaPushConstants {
    glm::vec2 inverse_extent{0.0F};
    float edge_threshold = 0.0F;
    float edge_threshold_min = 0.0F;
    float subpixel_quality = 0.0F;
    std::int32_t enabled = 0;
};
static_assert(offsetof(FxaaPushConstants, inverse_extent) == 0);
static_assert(offsetof(FxaaPushConstants, edge_threshold) == 8);
static_assert(offsetof(FxaaPushConstants, edge_threshold_min) == 12);
static_assert(offsetof(FxaaPushConstants, subpixel_quality) == 16);
static_assert(offsetof(FxaaPushConstants, enabled) == 20);
static_assert(sizeof(FxaaPushConstants) <= gpu::GraphicsPipeline::k_push_constant_size);

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

// 720p left the 3D view around 716 px wide once the panels took their columns,
// which is thin for an editor and thinner still for a capture of one.
constexpr std::uint32_t k_initial_width = 1600;
constexpr std::uint32_t k_initial_height = 900;

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

// The one docked column, as a fraction of the window. It carries the scene
// tree over the selection's properties; everything else is a menu, an overlay
// or a dialog, and the rest of the window is the 3D view.
constexpr float k_right_column_fraction = 0.22F;
// Inset of the read-out from the 3D view's top-right corner, and how opaque
// its backing is: enough to stay legible over a bright surface without hiding
// what is behind it.
constexpr float k_overlay_margin = 12.0F;
constexpr float k_overlay_alpha = 0.55F;

// Edits kept on the undo stack. Deep enough that no realistic session runs
// off the end, shallow enough that the closures cannot accumulate unboundedly.
constexpr std::size_t k_max_undo_depth = 128;

// Menu content has no panel to stretch into, so widgets that would otherwise
// fill the available width need one given to them.
constexpr float k_menu_item_width = 220.0F;
// The right column is split between the scene tree and the properties of
// whatever is selected in it.
constexpr float k_hierarchy_fraction = 0.5F;

// Fraction of the bounding sphere's fitted distance to back off by, so the
// model does not touch the edges of the view.
constexpr float k_framing_margin = 1.15F;

// Neutral, slightly rough dielectric. Bright enough to read against the dark
// background without being the brightest thing in frame.
constexpr float k_primitive_albedo = 0.8F;
constexpr float k_primitive_roughness = 0.6F;

// A new primitive's size, as a fraction of the current scene's radius. A plane
// is ground, so it wants to run past the edge of frame; a solid wants to look
// like an object sitting next to what is already there.
//
// The plane's factor is a trade rather than a taste: the shadow frustum is
// fitted to the whole scene, so a ground plane several times the model's size
// spends most of the shadow map on empty floor and coarsens the shadow on the
// model itself. Three is about where a floor still reads as a floor.
constexpr float k_plane_scene_fraction = 3.0F;
// Smaller than it first looks it should be, because a ground plane inflates
// the scene's diagonal that this is measured against: add a plane and then a
// sphere, and the sphere is sized against a scene the plane just made half as
// big again. A third keeps both orders sensible.
constexpr float k_solid_scene_fraction = 0.3F;

// A light icon marks a position; it is not an object in the scene, so it is a
// good deal smaller than a primitive added deliberately. Small enough to read
// as a marker rather than as geometry, but still a comfortable click target at
// the distances the camera frames a scene from.
constexpr float k_light_icon_fraction = 0.020F;
// A point light's reach, as a fraction of the scene. Beyond this the windowed
// falloff takes it to zero.
constexpr float k_point_range_fraction = 2.0F;
// A directional light's intensity is irradiance and does not fall off. A point
// light's divides by distance squared, so it needs far more to read at all.
constexpr float k_default_key_intensity = 8.0F;
constexpr float k_default_point_intensity = 60.0F;

// Mostly along -X, with far less -Z than the obvious diagonal.
//
// The obvious diagonal is what this was, and it is wrong for a default: the
// camera starts at +X, +Y, +Z and framing puts it back there, so a light
// shining along -X, -Y, -Z travels straight down the view axis. Its shadow
// falls directly behind the object, where the object itself hides it -- which
// looks exactly like shadows being broken. Casting across the view instead
// puts the shadow beside the subject, where it can be seen.
const glm::vec3 k_default_key_direction{-0.80F, -0.50F, -0.33F};

// Where the default light starts, before anything is loaded to measure. Only
// its icon's position; a directional light shades by rotation alone.
constexpr float k_default_light_height = 2.0F;
// Once there is a scene: how far back along the beam the marker sits, and the
// clearance above the scene's top it is never allowed below, both as fractions
// of the scene radius.
constexpr float k_default_light_distance = 2.0F;
constexpr float k_default_light_clearance = 0.3F;

constexpr float k_pi_f = std::numbers::pi_v<float>;

// A rotation whose -Y axis lands on `direction`, built as a basis rather than
// through glm's quaternion helpers: those live in GLM_GTX, which needs
// GLM_ENABLE_EXPERIMENTAL defined across every translation unit that includes
// glm. That is a project-wide decision to take for one rotation, and this is
// four lines.
glm::mat4 orientation_pointing_down_along(const glm::vec3& direction) {
    const glm::vec3 down = glm::normalize(direction);
    // The node's +Y is the opposite of where the light shines.
    const glm::vec3 up = -down;
    // Any hint not parallel to up; world X fails only when the light points
    // along X, which world Z then covers.
    const glm::vec3 hint =
        std::abs(up.x) > 0.99F ? glm::vec3(0.0F, 0.0F, 1.0F) : glm::vec3(1.0F, 0.0F, 0.0F);
    const glm::vec3 right = glm::normalize(glm::cross(hint, up));
    const glm::vec3 forward = glm::cross(up, right);
    return glm::mat4(glm::vec4(right, 0.0F), glm::vec4(up, 0.0F), glm::vec4(forward, 0.0F),
                     glm::vec4(0.0F, 0.0F, 0.0F, 1.0F));
}

// Where a context-menu placement lands when the cursor ray never meets the
// ground plane -- looking up, or along it. Far enough to be in front of the
// camera rather than inside it, near enough to stay in frame.
constexpr float k_fallback_placement_distance = 5.0F;
// Past this, a ray that technically does hit the ground is hitting it so far
// away that the object would be invisible. Treated as a miss.
constexpr float k_max_placement_distance = 1000.0F;

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
      ldr_target_(allocator_, device_, swapchain_.extent()),
      shadow_map_(allocator_, device_),
      pipeline_cache_(device_, pipeline_cache_path()),
      shadow_pipeline_(device_,
                       gpu::GraphicsPipelineDesc{
                           .spirv_path = std::filesystem::path(SAGE_SHADER_DIR) / "shadow.spv",
                           // No colour at all; depth is the entire output.
                           .depth_format = gpu::ShadowMap::format(),
                           .depth_only = true,
                           .depth_bias = true,
                           .set_layout = bindless_set_.layout(),
                           .cache = pipeline_cache_.handle(),
                       }),
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
                            // The LDR target, not the swapchain: FXAA sits
                            // between them now.
                            .color_format = gpu::LdrTarget::format(),
                            .cull_backfaces = false,
                            .set_layout = bindless_set_.layout(),
                            .cache = pipeline_cache_.handle(),
                        }),
      fxaa_pipeline_(device_,
                     gpu::GraphicsPipelineDesc{
                         .spirv_path = std::filesystem::path(SAGE_SHADER_DIR) / "fxaa.spv",
                         .color_format = swapchain_.format(),
                         .cull_backfaces = false,
                         .set_layout = bindless_set_.layout(),
                         .cache = pipeline_cache_.handle(),
                     }),
      frame_pacer_(device_),
      imgui_(instance_, device_, window_, swapchain_.format(), swapchain_.image_count()) {
    bindless_set_.write_object_id_image(id_buffer_.view());
    bindless_set_.write_hdr_color_image(hdr_target_.view());
    // Written once, unlike the two above: the shadow map does not follow the
    // swapchain, so a resize never invalidates this view.
    bindless_set_.write_shadow_map(shadow_map_.view(), shadow_map_.sampler());
    bindless_set_.write_ldr_color_image(ldr_target_.view(), ldr_target_.sampler());
    gizmo_operation_ = static_cast<int>(ImGuizmo::TRANSLATE);
    tonemap_operator_ = static_cast<int>(TonemapOperator::aces);
    if (!gpu::is_capturable_format(swapchain_.format())) {
        SAGE_LOG_WARN("Swapchain format {} cannot be captured; screenshots are disabled",
                      static_cast<int>(swapchain_.format()));
    }

    if (model_path.empty()) {
        // Nothing will call clear_scene, so the default light is this path's to
        // create. With a model named, load_model clears first and clear_scene
        // puts one back -- doing it here as well would build one and throw it
        // away a line later.
        add_default_light();
        SAGE_LOG_INFO("No model given; use the Load glTF panel to pick one");
        return;
    }

    // Through the same path a picker click takes, so a command-line model gets
    // no special handling and a bad argument is a message rather than a crash
    // before the window is ever useful.
    if (!load_model(PendingLoad{model_path, true, std::nullopt})) {
        SAGE_LOG_WARN("Starting with an empty scene");
    }
}

bool Application::load_model(const PendingLoad& load) {
    if (load.replace) {
        clear_scene();
    }

    const std::optional<gpu::LoadedScene> loaded = gpu::load_gltf(
        load.path, geometry_registry_, texture_registry_, material_registry_, scene_graph_);
    if (!loaded.has_value()) {
        return false;
    }

    // Everything from a file hangs off one root, so placing the load is one
    // transform rather than a walk.
    if (load.placement.has_value()) {
        scene_graph_.set_local_transform(loaded->root,
                                         glm::translate(glm::mat4(1.0F), *load.placement));
        scene_graph_.update_transforms();
    }

    // Only when the load actually drew something and had the viewport to
    // itself. Re-framing on an additive load would yank the camera away from
    // whatever the user was looking at to fit a model they just added.
    //
    // Queued rather than applied: framing needs the central node's aspect, and
    // the dockspace does not exist yet when the constructor loads a model named
    // on the command line.
    if (load.replace && loaded->mesh_count > 0) {
        pending_frame_ = SceneBounds{loaded->bounds_min, loaded->bounds_max};
    }

    // Now that there is something to measure. Also runs for an additive load,
    // where the scene has grown and a marker sized for the old one is in the
    // wrong place -- but only while the light is still untouched.
    reposition_default_light();

    // Only an additive load. A replacing one ran clear_scene on the way in,
    // which rewound the registries and dropped the history with them: there is
    // no previous scene left to go back to.
    if (!load.replace) {
        push_add_command("Add " + load.path.filename().string(), loaded->root, selected_);
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

    // The tree's open/closed state is keyed on node index, and indices restart
    // at zero after this. Without dropping it, a replace-load would inherit
    // whatever the previous scene had been expanded to -- so the next file
    // would come up part-opened on nodes that have nothing to do with the ones
    // that were opened. Deferred because the storage belongs to the Hierarchy
    // window, which is only current inside its own Begin/End.
    hierarchy_state_stale_ = true;

    // Every handle on the stack has just gone stale, and the registries have
    // rewound, so nothing here could be put back even if the handles survived.
    // Dropping the history is the honest answer; a Ctrl+Z that silently did
    // nothing would be worse.
    clear_history();

    // A light is a node, so clearing the graph removed it. Without putting one
    // back, the next model would load into a scene with nothing lighting it.
    add_default_light();
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

    const PendingLoad load = *pending_load_;
    pending_load_.reset();

    if (!load_model(load)) {
        SAGE_LOG_ERROR("Could not load {}", load.path.string());
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

std::uint32_t Application::collect_lights(std::array<gpu::Light, gpu::k_max_lights>& lights) const {
    std::uint32_t count = 0;

    for (const gpu::SceneNode& node : scene_graph_.nodes()) {
        if (!node.alive || !node.has_light) {
            continue;
        }
        if (count >= gpu::k_max_lights) {
            // Dropped rather than grown: the frame buffer's light array is a
            // fixed size the shader also declares, so the limit is a layout
            // fact rather than a policy this function can bend.
            break;
        }

        gpu::Light& light = lights[count];
        light.type = node.light.type;
        light.color = node.light.color;
        light.intensity = node.light.intensity;
        light.range = node.light.range;
        // Both derived from the transform, which is what makes the gizmo work
        // on a light at all.
        light.position = glm::vec3(node.world_transform[3]);
        // -Y is the canonical direction, so an unrotated light points down and
        // the rotate gizmo tilts it from there. Normalised because a scaled
        // node would otherwise hand the shader a non-unit direction, which the
        // BRDF has no way to notice and every dot product would be wrong by.
        const glm::vec3 down = glm::mat3(node.world_transform) * glm::vec3(0.0F, -1.0F, 0.0F);
        const float length = glm::length(down);
        light.direction = length > 1e-6F ? down / length : glm::vec3(0.0F, -1.0F, 0.0F);
        ++count;
    }
    return count;
}

gpu::NodeHandle Application::first_directional_light() const {
    for (std::size_t i = 0; i < scene_graph_.nodes().size(); ++i) {
        const gpu::SceneNode& node = scene_graph_.nodes()[i];
        if (node.alive && node.has_light && node.light.type == gpu::LightType::directional) {
            return scene_graph_.handle_at(i);
        }
    }
    return gpu::NodeHandle{};
}

Application::Bounds Application::scene_bounds() const {
    Bounds bounds{glm::vec3(std::numeric_limits<float>::max()),
                  glm::vec3(std::numeric_limits<float>::lowest())};

    for (const gpu::SceneNode& node : scene_graph_.nodes()) {
        // editor_only excluded as well as mesh-less. A light icon is not scene
        // content, and counting it here would cost three separate things: the
        // shadow frustum would stretch to cover a marker floating above the
        // subject and spend its resolution on empty air, a new primitive would
        // be sized against a scene the markers made bigger, and repositioning
        // the default light would move the very icon that set the bounds it
        // was positioned from.
        if (!node.alive || !node.has_mesh || node.editor_only) {
            continue;
        }
        const glm::vec3& local_min = node.mesh.bounds_min;
        const glm::vec3& local_max = node.mesh.bounds_max;

        // All eight corners, not just the two. Transforming min and max alone
        // is only correct for an axis-aligned transform; under any rotation it
        // yields a box that does not contain the mesh.
        for (int corner = 0; corner < 8; ++corner) {
            const glm::vec3 local{(corner & 1) != 0 ? local_max.x : local_min.x,
                                  (corner & 2) != 0 ? local_max.y : local_min.y,
                                  (corner & 4) != 0 ? local_max.z : local_min.z};
            const glm::vec3 world = glm::vec3(node.world_transform * glm::vec4(local, 1.0F));
            bounds.min = glm::min(bounds.min, world);
            bounds.max = glm::max(bounds.max, world);
        }
    }
    return bounds;
}

Application::LightFit Application::fit_light(const Bounds& bounds,
                                             const glm::vec3& light_direction) const {
    if (bounds.empty()) {
        return {};
    }

    const glm::vec3 center = (bounds.min + bounds.max) * 0.5F;
    // The bounding sphere, not the box. A box's extent depends on how it is
    // turned; fitting the sphere means the frustum stays the same size as the
    // light swings around, so the shadow's resolution does not change while
    // the direction slider is dragged.
    const float radius = glm::length(bounds.max - bounds.min) * 0.5F;
    // A degenerate scene -- one point, or nothing -- would make a zero-extent
    // projection and divide by zero.
    const float extent = std::max(radius, 1e-3F);

    const glm::vec3 direction = glm::normalize(light_direction);

    // Any vector not parallel to the light will do as an up hint; world up
    // fails exactly when the light points straight down, which is a common
    // enough setting to handle rather than hope about.
    const glm::vec3 up =
        std::abs(direction.y) > 0.99F ? glm::vec3(0.0F, 0.0F, 1.0F) : glm::vec3(0.0F, 1.0F, 0.0F);

    // Pulled back a full diameter beyond the sphere so that geometry behind
    // the centre still falls inside the near plane and casts.
    const glm::vec3 eye = center - direction * (extent * 2.0F);
    const glm::mat4 view = glm::lookAt(eye, center, up);

    // Depth range covers the pull-back plus the far side of the sphere. Kept
    // tight rather than generous: every unit of depth range spent on empty
    // space is precision taken from the part that has geometry in it.
    const glm::mat4 projection =
        core::ortho_vk(-extent, extent, -extent, extent, 0.0F, extent * 4.0F);

    LightFit fit;
    fit.view_projection = projection * view;
    // The frustum spans 2 * extent across `resolution` texels.
    fit.world_texel_size = (extent * 2.0F) / static_cast<float>(shadow_map_.resolution());
    return fit;
}

void Application::write_frame_data(std::uint32_t frame_slot) const {
    const CameraMatrices matrices = camera_matrices();

    // One write per frame, into this frame's own slot. Writing a single shared
    // slot would race the GPU, which may still be reading the previous frame's
    // copy. begin_frame() has already waited out the work that used this slot.
    FrameData frame_data;
    frame_data.view_projection = matrices.projection * matrices.view;
    frame_data.camera_position = camera_.position();

    // Gathered from the graph rather than from members. The shader has always
    // read lights as data; what changed is that the data now comes from nodes,
    // so a light can be placed, parented and dragged like anything else.
    frame_data.light_count = collect_lights(frame_data.lights);
    frame_data.ambient_intensity = ambient_intensity_;

    // Refitted every frame from live world transforms. A gizmo drag moves
    // geometry, which moves the bounds, which moves the frustum -- so a shadow
    // keeps up with the thing casting it.
    // The shadow map is fitted to one directional light -- the first in the
    // graph, matching the shader, which spends it on the first directional
    // light it evaluates. With none in the scene there is nothing to fit and
    // nothing to cast, so the pass still runs but the lookup is skipped.
    const gpu::SceneNode* key = scene_graph_.find(first_directional_light());
    const glm::vec3 key_direction =
        key != nullptr ? glm::vec3(glm::mat3(key->world_transform) * glm::vec3(0.0F, -1.0F, 0.0F))
                       : glm::vec3(0.0F, -1.0F, 0.0F);

    const LightFit fit = fit_light(scene_bounds(), key_direction);
    frame_data.light_view_projection = fit.view_projection;
    // Texels converted to world units here, so the shader stays in world space
    // and the slider keeps meaning the same thing at any scene scale.
    frame_data.shadow_normal_bias = shadow_normal_bias_texels_ * fit.world_texel_size;
    frame_data.shadow_pcf_radius = shadow_pcf_radius_;
    frame_data.shadow_texel_size = 1.0F / static_cast<float>(shadow_map_.resolution());
    frame_data.shadow_enabled = (shadows_enabled_ && key != nullptr) ? 1U : 0U;

    frame_buffer_.write(&frame_data, sizeof(frame_data),
                        VkDeviceSize{frame_slot} * sizeof(FrameData));
}

void Application::record_shadow(VkCommandBuffer command_buffer, std::uint32_t frame_slot) const {
    VkImageMemoryBarrier2 to_depth{};
    to_depth.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    // The previous frame's last use was the fragment shader sampling this map,
    // so that read has to finish before the pass below overwrites it. A
    // write-after-read, hence no source access mask.
    to_depth.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    to_depth.srcAccessMask = VK_ACCESS_2_NONE;
    to_depth.dstStageMask =
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    to_depth.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    to_depth.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_depth.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    to_depth.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_depth.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_depth.image = shadow_map_.image();
    to_depth.subresourceRange = k_depth_range;

    VkDependencyInfo begin_dependency{};
    begin_dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    begin_dependency.imageMemoryBarrierCount = 1;
    begin_dependency.pImageMemoryBarriers = &to_depth;
    vkCmdPipelineBarrier2(command_buffer, &begin_dependency);

    // STORE, unlike the main depth buffer's DONT_CARE: this attachment's whole
    // purpose is to be read back afterwards.
    VkRenderingAttachmentInfo depth_attachment{};
    depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_attachment.imageView = shadow_map_.view();
    depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.clearValue.depthStencil = {1.0F, 0};

    const VkExtent2D extent = shadow_map_.extent();

    VkRenderingInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering.renderArea.offset = {0, 0};
    rendering.renderArea.extent = extent;
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 0;
    rendering.pDepthAttachment = &depth_attachment;

    vkCmdBeginRendering(command_buffer, &rendering);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_.handle());

    const VkDescriptorSet descriptor_set = bindless_set_.handle();
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            shadow_pipeline_.layout(), 0, 1, &descriptor_set, 0, nullptr);

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

    // Dynamic, so the two ends of the bias trade-off stay draggable. The
    // constant term is in units of the depth format's smallest representable
    // difference; the slope term scales with how steeply the surface is turned
    // away from the light, which is where acne appears first.
    vkCmdSetDepthBias(command_buffer, shadow_depth_bias_, 0.0F, shadow_slope_bias_);

    const VkDeviceAddress frame_address =
        frame_buffer_.device_address() + (VkDeviceSize{frame_slot} * sizeof(FrameData));

    for (const gpu::SceneNode& node : scene_graph_.nodes()) {
        // editor_only skipped as well as mesh-less: a marker standing for a
        // light must not cast a shadow of its own.
        if (!node.alive || !node.has_mesh || node.editor_only) {
            continue;
        }
        // The same struct the main pass pushes. This pipeline's shader declares
        // only the first three members and ignores the rest.
        PushConstants push{};
        push.vertex_address = node.mesh.vertex_address;
        push.frame_address = frame_address;
        push.model = node.world_transform;
        vkCmdPushConstants(command_buffer, shadow_pipeline_.layout(), VK_SHADER_STAGE_ALL, 0,
                           sizeof(push), &push);
        vkCmdBindIndexBuffer(command_buffer, geometry_registry_.buffer(), node.mesh.index_offset,
                             VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(command_buffer, node.mesh.index_count, 1, 0, 0, 0);
    }

    vkCmdEndRendering(command_buffer);

    // Into the layout the descriptor written at startup names. DEPTH_READ_ONLY
    // rather than SHADER_READ_ONLY because this is a depth image.
    VkImageMemoryBarrier2 to_read{};
    to_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_read.srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    to_read.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    to_read.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    to_read.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    to_read.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    to_read.newLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
    to_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_read.image = shadow_map_.image();
    to_read.subresourceRange = k_depth_range;

    VkDependencyInfo end_dependency{};
    end_dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    end_dependency.imageMemoryBarrierCount = 1;
    end_dependency.pImageMemoryBarriers = &to_read;
    vkCmdPipelineBarrier2(command_buffer, &end_dependency);
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

    // Already written by write_frame_data, before the shadow pass -- both
    // passes read the same slot, and the shadow pass needs the light matrix
    // that is in it.
    const VkDeviceAddress frame_address =
        frame_buffer_.device_address() + (VkDeviceSize{frame_slot} * sizeof(FrameData));

    // Light icons are drawn with the scene so they pick and outline like any
    // other mesh, but they are the tool rather than the render: hidden with the
    // panels, and hidden for the frame a no-UI capture is taken on. That frame
    // is also the one on screen, so pressing F2 without F11 blinks them for a
    // single frame -- the alternative is rendering the scene twice.
    const bool show_editor_meshes =
        !ui_hidden_ && !(pending_screenshot_.has_value() && !screenshot_include_ui_);

    for (std::uint32_t index = 0; index < scene_graph_.nodes().size(); ++index) {
        const gpu::SceneNode& node = scene_graph_.nodes()[index];
        if (!node.alive || !node.has_mesh) {
            // Deleted, or a pure transform node: glTF hierarchy nodes, and the
            // per-load root.
            continue;
        }
        if (node.editor_only && !show_editor_meshes) {
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

void Application::record_tonemap(VkCommandBuffer command_buffer) const {
    // Two transitions, one dependency. The HDR target stops being an attachment
    // and becomes a texture; the LDR target, which the FXAA pass sampled last
    // frame, goes the other way.
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

    // Write-after-read: the previous frame's FXAA pass sampled this image, and
    // that read has to finish before the tonemap overwrites it.
    VkImageMemoryBarrier2 ldr_to_color{};
    ldr_to_color.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    ldr_to_color.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    ldr_to_color.srcAccessMask = VK_ACCESS_2_NONE;
    ldr_to_color.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    ldr_to_color.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    ldr_to_color.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ldr_to_color.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ldr_to_color.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ldr_to_color.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ldr_to_color.image = ldr_target_.image();
    ldr_to_color.subresourceRange = k_color_range;

    const std::array<VkImageMemoryBarrier2, 2> barriers{hdr_to_read, ldr_to_color};

    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.imageMemoryBarrierCount = static_cast<std::uint32_t>(barriers.size());
    dependency.pImageMemoryBarriers = barriers.data();
    vkCmdPipelineBarrier2(command_buffer, &dependency);

    // DONT_CARE, not CLEAR: the triangle below covers every pixel of the image,
    // so clearing first would be writing the whole target twice.
    VkRenderingAttachmentInfo color_attachment{};
    color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachment.imageView = ldr_target_.view();
    color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    const VkExtent2D extent = ldr_target_.extent();

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

void Application::record_fxaa(VkCommandBuffer command_buffer, VkImage image,
                              VkImageView image_view) const {
    // The LDR target stops being an attachment and becomes a texture; the
    // swapchain image, untouched so far this frame, becomes an attachment for
    // the first time.
    VkImageMemoryBarrier2 ldr_to_read{};
    ldr_to_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    ldr_to_read.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    ldr_to_read.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    ldr_to_read.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    ldr_to_read.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    ldr_to_read.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ldr_to_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ldr_to_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ldr_to_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ldr_to_read.image = ldr_target_.image();
    ldr_to_read.subresourceRange = k_color_range;

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

    const std::array<VkImageMemoryBarrier2, 2> barriers{ldr_to_read, swapchain_to_color};

    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.imageMemoryBarrierCount = static_cast<std::uint32_t>(barriers.size());
    dependency.pImageMemoryBarriers = barriers.data();
    vkCmdPipelineBarrier2(command_buffer, &dependency);

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
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, fxaa_pipeline_.handle());

    const VkDescriptorSet descriptor_set = bindless_set_.handle();
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            fxaa_pipeline_.layout(), 0, 1, &descriptor_set, 0, nullptr);

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

    FxaaPushConstants push{};
    push.inverse_extent = glm::vec2(1.0F / static_cast<float>(extent.width),
                                    1.0F / static_cast<float>(extent.height));
    push.edge_threshold = fxaa_edge_threshold_;
    push.edge_threshold_min = fxaa_edge_threshold_min_;
    push.subpixel_quality = fxaa_subpixel_quality_;
    push.enabled = fxaa_enabled_ ? 1 : 0;
    vkCmdPushConstants(command_buffer, fxaa_pipeline_.layout(), VK_SHADER_STAGE_ALL, 0,
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

    const ImVec2 mouse = ImGui::GetMousePos();
    pending_pick_ = viewport_texel(mouse.x, mouse.y);
}

std::optional<VkOffset2D> Application::viewport_texel(float window_x, float window_y) const {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    if (viewport->Size.x <= 0.0F || viewport->Size.y <= 0.0F) {
        return std::nullopt;
    }

    // ImGui reports logical coordinates; the attachments are framebuffer
    // pixels. Same conversion as central_node_rect, for the same reason.
    const VkExtent2D extent = swapchain_.extent();
    const float x =
        (window_x - viewport->Pos.x) * (static_cast<float>(extent.width) / viewport->Size.x);
    const float y =
        (window_y - viewport->Pos.y) * (static_cast<float>(extent.height) / viewport->Size.y);

    const auto texel_x = static_cast<std::int32_t>(x);
    const auto texel_y = static_cast<std::int32_t>(y);

    // Outside the 3D view is not a miss, it is not a click in it at all --
    // clicking the dockspace border should leave the selection alone, and
    // should not offer to add anything either.
    const VkRect2D& rect = viewport_rect_;
    if (texel_x < rect.offset.x || texel_y < rect.offset.y ||
        texel_x >= rect.offset.x + static_cast<std::int32_t>(rect.extent.width) ||
        texel_y >= rect.offset.y + static_cast<std::int32_t>(rect.extent.height)) {
        return std::nullopt;
    }
    return VkOffset2D{texel_x, texel_y};
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

    // Either the 3D view alone -- recorded before any overlay reaches the image
    // -- or the whole window after ImGui, which is a picture of the editor
    // rather than of the render.
    const VkRect2D region_rect =
        screenshot_include_ui_ ? VkRect2D{{0, 0}, swapchain_.extent()} : viewport_rect_;
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
    // With the UI hidden there is no dockspace at all, and the 3D view takes
    // the whole image. Set explicitly rather than by letting the central node
    // grow: a node whose windows were simply not submitted keeps whatever size
    // the layout gave it, so the rect would be a frame behind at best.
    if (ui_hidden_) {
        viewport_rect_ = VkRect2D{{0, 0}, swapchain_.extent()};
        const ImGuiViewport* whole = ImGui::GetMainViewport();
        viewport_logical_pos_ = glm::vec2(whole->Pos.x, whole->Pos.y);
        viewport_logical_size_ = glm::vec2(whole->Size.x, whole->Size.y);
        return;
    }

    // PassthruCentralNode leaves the middle node transparent and, while it is
    // empty, lets mouse input through it -- so the scene shows and the camera
    // still responds. NoDockingOverCentralNode keeps it empty permanently, so a
    // panel cannot be dragged over the 3D view by accident.
    const ImGuiID dockspace = ImGui::DockSpaceOverViewport(
        0, ImGui::GetMainViewport(),
        ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_NoDockingOverCentralNode);

    viewport_rect_ = central_node_rect(dockspace, swapchain_.extent());
    update_viewport_logical_rect(dockspace);

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

    // One column, on the right. The left one is gone: the read-out it held is
    // now an overlay inside the 3D view and the browser it held is a dialog off
    // the File menu, so a whole column of screen was being spent on two things
    // that needed no permanent home.
    ImGuiID right = 0;
    ImGuiID centre = 0;
    ImGui::DockBuilderSplitNode(dockspace, ImGuiDir_Right, k_right_column_fraction, &right,
                                &centre);

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
    update_viewport_logical_rect(dockspace);
}

void Application::update_viewport_logical_rect(unsigned int dockspace) {
    const ImGuiDockNode* central = ImGui::DockBuilderGetCentralNode(dockspace);
    const ImGuiViewport* whole = ImGui::GetMainViewport();
    if (central == nullptr || central->Size.x <= 0.0F || central->Size.y <= 0.0F) {
        viewport_logical_pos_ = glm::vec2(whole->Pos.x, whole->Pos.y);
        viewport_logical_size_ = glm::vec2(whole->Size.x, whole->Size.y);
        return;
    }
    viewport_logical_pos_ = glm::vec2(central->Pos.x, central->Pos.y);
    viewport_logical_size_ = glm::vec2(central->Size.x, central->Size.y);
}

void Application::draw_stats_overlay() {
    // Pinned inside the 3D view rather than to the window, so it tracks the
    // central node as panels resize and follows the whole screen once they are
    // hidden.
    const ImVec2 corner{viewport_logical_pos_.x + viewport_logical_size_.x - k_overlay_margin,
                        viewport_logical_pos_.y + k_overlay_margin};
    ImGui::SetNextWindowPos(corner, ImGuiCond_Always, ImVec2(1.0F, 0.0F));
    ImGui::SetNextWindowBgAlpha(k_overlay_alpha);

    // NoInputs is the one that matters: without it the overlay would swallow
    // clicks meant for whatever is behind it, and picking would go dead in one
    // corner of the viewport for no visible reason.
    const ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDocking |
        ImGuiWindowFlags_NoInputs;

    ImGui::Begin("##stats", nullptr, flags);
    const ImGuiIO& io = ImGui::GetIO();
    ImGui::Text("%.1f fps  %.2f ms", static_cast<double>(io.Framerate),
                1000.0 / static_cast<double>(io.Framerate));
    ImGui::Separator();
    ImGui::Text("%zu nodes", scene_graph_.live_size());
    ImGui::Text("Geometry  %llu / %llu KiB",
                static_cast<unsigned long long>(geometry_registry_.used() / 1024),
                static_cast<unsigned long long>(geometry_registry_.capacity() / 1024));
    ImGui::Text("Materials %u / %u", material_registry_.count(), material_registry_.capacity());
    ImGui::Text("Textures  %u", texture_registry_.count());
    ImGui::Separator();
    const gpu::SceneNode* selected = scene_graph_.find(selected_);
    ImGui::Text("Selected  %s", selected != nullptr ? selected->name.c_str() : "(none)");
    const glm::vec3 position = camera_.position();
    ImGui::Text("Camera    %.1f, %.1f, %.1f", static_cast<double>(position.x),
                static_cast<double>(position.y), static_cast<double>(position.z));
    ImGui::End();
}

void Application::draw_menu_bar() {
    if (!ImGui::BeginMainMenuBar()) {
        return;
    }

    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Open glTF...")) {
            file_dialog_open_ = true;
            file_dialog_scene_actions_ = true;
            // Absent, so the file lands where it says it does. Only the context
            // menu places a load.
            file_dialog_placement_.reset();
        }
        if (ImGui::MenuItem("Clear scene")) {
            // Queued: clear_scene waits for the device to go idle and destroys
            // images a recording command buffer still names.
            pending_clear_ = true;
        }
        ImGui::Separator();
        ImGui::MenuItem("Include UI in captures", nullptr, &screenshot_include_ui_);
        if (ImGui::MenuItem("Screenshot", "F2", false, !pending_screenshot_.has_value())) {
            request_screenshot();
        }
        // What the next capture will actually contain, which the toggle above
        // decides and is otherwise only discoverable by taking one.
        if (screenshot_include_ui_) {
            ImGui::TextDisabled("%ux%u, whole window", swapchain_.extent().width,
                                swapchain_.extent().height);
        } else {
            ImGui::TextDisabled("%ux%u, no UI", viewport_rect_.extent.width,
                                viewport_rect_.extent.height);
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Edit")) {
        // Labelled with what they would actually reverse, so the menu says
        // "Undo Transform" rather than leaving you to remember what you did.
        const std::string undo_label =
            can_undo() ? "Undo " + undo_stack_[undo_cursor_ - 1].name : std::string("Undo");
        const std::string redo_label =
            can_redo() ? "Redo " + undo_stack_[undo_cursor_].name : std::string("Redo");
        if (ImGui::MenuItem(undo_label.c_str(), "Ctrl+Z", false, can_undo())) {
            undo();
        }
        if (ImGui::MenuItem(redo_label.c_str(), "Ctrl+Y", false, can_redo())) {
            redo();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Delete", "Del", false, scene_graph_.find(selected_) != nullptr)) {
            delete_selected();
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Render")) {
        draw_presentation_controls();
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Lighting")) {
        draw_lighting_menu();
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("View")) {
        ImGui::MenuItem("Hide panels", "F11", &ui_hidden_);
        ImGui::EndMenu();
    }

    ImGui::EndMainMenuBar();
}

void Application::draw_file_dialog() {
    if (file_dialog_open_ && !ImGui::IsPopupOpen("Load glTF")) {
        ImGui::OpenPopup("Load glTF");
    }

    // Centred rather than at the cursor: the browser is a good deal larger than
    // a menu, and anchoring it to a click near an edge would push it off-screen.
    const ImVec2 centre = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(centre, ImGuiCond_Appearing, ImVec2(0.5F, 0.5F));
    if (!ImGui::BeginPopupModal("Load glTF", &file_dialog_open_,
                                ImGuiWindowFlags_AlwaysAutoResize)) {
        return;
    }

    if (const std::optional<FilePicker::Request> request =
            file_picker_.draw_contents(file_dialog_scene_actions_);
        request.has_value()) {
        // Queued rather than loaded here: this is the middle of a frame, and a
        // load blocks, waits for the device and destroys images the command
        // buffer being recorded would still reference.
        pending_load_ = PendingLoad{request->path, request->replace, file_dialog_placement_};
        file_dialog_open_ = false;
        ImGui::CloseCurrentPopup();
    }
    if (file_picker_.clear_requested()) {
        pending_clear_ = true;
        file_dialog_open_ = false;
        ImGui::CloseCurrentPopup();
    }
    if (ImGui::Button("Cancel")) {
        file_dialog_open_ = false;
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

void Application::draw_presentation_controls() {
    ImGui::SeparatorText("Tonemap");

    // Selectable at runtime rather than baked in, because the difference between
    // two curves is only legible on the same frame -- comparing across a rebuild
    // compares two memories of an image.
    ImGui::SetNextItemWidth(k_menu_item_width);
    ImGui::Combo("##tonemap", &tonemap_operator_, k_tonemap_names.data(),
                 static_cast<int>(k_tonemap_names.size()));
    if (tonemap_operator_ == static_cast<int>(TonemapOperator::none)) {
        ImGui::TextDisabled("Clamped, as before M8");
    }

    ImGui::SliderFloat("Exposure", &exposure_stops_, k_min_exposure_stops, k_max_exposure_stops,
                       "%+.2f stops");

    ImGui::SeparatorText("Anti-aliasing");
    ImGui::Checkbox("FXAA", &fxaa_enabled_);
    ImGui::BeginDisabled(!fxaa_enabled_);
    // Lower catches more edges and softens more of the image with them; higher
    // leaves faint edges alone. The floor matters most in dark regions, where a
    // tiny absolute difference is a large relative one.
    ImGui::SliderFloat("Edge threshold", &fxaa_edge_threshold_, 0.03F, 0.33F, "%.3f");
    ImGui::SliderFloat("Dark floor", &fxaa_edge_threshold_min_, 0.005F, 0.1F, "%.4f");
    ImGui::SliderFloat("Subpixel", &fxaa_subpixel_quality_, 0.0F, 1.0F, "%.2f");
    ImGui::EndDisabled();
}

bool Application::add_primitive(gpu::PrimitiveKind kind, const glm::vec3& position) {
    const gpu::PrimitiveMesh mesh = gpu::make_primitive(kind);

    glm::vec3 low{std::numeric_limits<float>::max()};
    glm::vec3 high{std::numeric_limits<float>::lowest()};
    for (const gpu::Vertex& vertex : mesh.vertices) {
        low = glm::min(low, vertex.position);
        high = glm::max(high, vertex.position);
    }

    const gpu::GeometryRegistry::MeshView view = geometry_registry_.add_mesh(
        mesh.vertices.data(), sizeof(gpu::Vertex) * mesh.vertices.size(), mesh.indices.data(),
        static_cast<std::uint32_t>(mesh.indices.size()), low, high);
    if (!view.valid()) {
        return false;
    }

    // Its own material rather than one shared table entry. A shared one would
    // have to be rebuilt after every clear_scene, since the registry rewinds,
    // and giving each primitive its own leaves room to tint them individually.
    gpu::Material material;
    material.base_color_factor =
        glm::vec4(k_primitive_albedo, k_primitive_albedo, k_primitive_albedo, 1.0F);
    material.metallic = 0.0F;
    material.roughness = k_primitive_roughness;
    // The registry's two permanent fallbacks: white multiplies to the factor,
    // and the flat normal decodes to the geometric one.
    material.base_color_texture = gpu::TextureRegistry::k_fallback_slot;
    material.normal_texture = gpu::TextureRegistry::k_flat_normal_slot;
    material.metallic_roughness_texture = gpu::TextureRegistry::k_fallback_slot;
    material.emissive_texture = gpu::TextureRegistry::k_fallback_slot;
    const std::uint32_t material_index = material_registry_.append({material});

    // Sized against what is already here. A fixed size cannot suit both a
    // chess piece and a street lamp -- the sample models alone span a factor
    // of 36 -- so an absolute default would be wrong for nearly every scene.
    const Bounds bounds = scene_bounds();
    const float radius = bounds.empty() ? 0.0F : glm::length(bounds.max - bounds.min) * 0.5F;
    const float reference = radius > 1e-4F ? radius : 1.0F;
    const float scale =
        (kind == gpu::PrimitiveKind::plane ? k_plane_scene_fraction : k_solid_scene_fraction) *
        reference;

    // Solids rest on the placement point rather than being buried half in it;
    // the plane is the ground, so it sits exactly there.
    const float lift = kind == gpu::PrimitiveKind::plane ? 0.0F : -low.y * scale;

    glm::mat4 transform = glm::translate(glm::mat4(1.0F), position + glm::vec3(0.0F, lift, 0.0F));
    transform = glm::scale(transform, glm::vec3(scale));

    const gpu::NodeHandle previous_selection = selected_;
    const gpu::NodeHandle node =
        scene_graph_.add_node(gpu::NodeHandle{}, transform, gpu::primitive_name(kind));
    scene_graph_.set_mesh(node, view, material_index);
    scene_graph_.update_transforms();
    select(node);
    push_add_command(std::string("Add ") + gpu::primitive_name(kind), node, previous_selection);

    SAGE_LOG_INFO("Added {} at ({:.3f}, {:.3f}, {:.3f}), scale {:.3f}", gpu::primitive_name(kind),
                  position.x, position.y, position.z, scale);
    return true;
}

bool Application::add_light(gpu::LightType type, const glm::vec3& position, bool record) {
    const bool directional = type == gpu::LightType::directional;

    gpu::SceneLight authored;
    authored.type = type;
    authored.color = glm::vec3(1.0F, 0.96F, 0.9F);
    // A directional light's intensity is irradiance and does not fall off; a
    // point light's is divided by distance squared, so the same number would
    // be invisible a couple of units away.
    authored.intensity = directional ? k_default_key_intensity : k_default_point_intensity;

    const Bounds bounds = scene_bounds();
    const float radius = bounds.empty() ? 1.0F : glm::length(bounds.max - bounds.min) * 0.5F;
    const float reference = radius > 1e-4F ? radius : 1.0F;
    authored.range = reference * k_point_range_fraction;

    // Rotated so -Y, the canonical direction, points the way the default key
    // light did. A directional light has no position, but the node still needs
    // one to put its icon somewhere and to give the gizmo something to hold.
    glm::mat4 transform = glm::translate(glm::mat4(1.0F), position);
    if (directional) {
        transform *= orientation_pointing_down_along(k_default_key_direction);
    }

    const gpu::NodeHandle previous_selection = selected_;
    const gpu::NodeHandle node = scene_graph_.add_node(
        gpu::NodeHandle{}, transform, directional ? "Directional Light" : "Point Light");
    scene_graph_.set_light(node, authored);

    // A light has no geometry, so without this it could not be picked in the
    // viewport, could not be outlined, and would be reachable only from the
    // hierarchy -- which is exactly when you least want to leave the 3D view.
    const gpu::PrimitiveMesh icon =
        gpu::make_primitive(directional ? gpu::PrimitiveKind::cone : gpu::PrimitiveKind::sphere);
    glm::vec3 low{std::numeric_limits<float>::max()};
    glm::vec3 high{std::numeric_limits<float>::lowest()};
    for (const gpu::Vertex& vertex : icon.vertices) {
        low = glm::min(low, vertex.position);
        high = glm::max(high, vertex.position);
    }

    const gpu::GeometryRegistry::MeshView view = geometry_registry_.add_mesh(
        icon.vertices.data(), sizeof(gpu::Vertex) * icon.vertices.size(), icon.indices.data(),
        static_cast<std::uint32_t>(icon.indices.size()), low, high);
    if (!view.valid()) {
        return false;
    }

    // Emissive rather than lit: an icon standing for a light source should
    // read as one, and a shaded grey blob sitting in mid-air reads as an
    // object that someone forgot to delete.
    gpu::Material material;
    material.base_color_factor = glm::vec4(0.0F, 0.0F, 0.0F, 1.0F);
    material.emissive_factor = authored.color;
    material.metallic = 0.0F;
    material.roughness = 1.0F;
    material.base_color_texture = gpu::TextureRegistry::k_fallback_slot;
    material.normal_texture = gpu::TextureRegistry::k_flat_normal_slot;
    material.metallic_roughness_texture = gpu::TextureRegistry::k_fallback_slot;
    material.emissive_texture = gpu::TextureRegistry::k_fallback_slot;
    const std::uint32_t material_index = material_registry_.append({material});

    // A child, so moving the light moves its icon and the icon never needs
    // updating separately. Scaled small: it marks a position, it is not a
    // thing in the scene.
    const float icon_scale = reference * k_light_icon_fraction;
    glm::mat4 icon_transform = glm::scale(glm::mat4(1.0F), glm::vec3(icon_scale));
    if (directional) {
        // Cone tip towards -Y, so it reads as an arrow pointing the way the
        // light travels. The generated cone points +Y.
        icon_transform =
            glm::rotate(glm::mat4(1.0F), k_pi_f, glm::vec3(1.0F, 0.0F, 0.0F)) * icon_transform;
    }

    const gpu::NodeHandle icon_node = scene_graph_.add_node(node, icon_transform, "Icon");
    scene_graph_.set_mesh(icon_node, view, material_index);
    scene_graph_.set_editor_only(icon_node, true);
    scene_graph_.update_transforms();
    select(node);
    if (record) {
        push_add_command(directional ? "Add directional light" : "Add point light", node,
                         previous_selection);
    }

    SAGE_LOG_INFO("Added {} at ({:.3f}, {:.3f}, {:.3f})",
                  directional ? "directional light" : "point light", position.x, position.y,
                  position.z);
    return true;
}

void Application::add_default_light() {
    // The scene is almost always empty here -- clear_scene calls this before
    // anything is loaded -- so this position is a placeholder that
    // reposition_default_light replaces once there is something to measure.
    // Not recorded. The default light is setup rather than an edit: putting it
    // on the stack means the very first Ctrl+Z in a fresh scene deletes the
    // only light and leaves the viewport dark, undoing something the user
    // never did.
    static_cast<void>(add_light(gpu::LightType::directional,
                                glm::vec3(0.0F, k_default_light_height, 0.0F), false));
    // add_light selects what it adds, which is how its handle is recovered
    // without threading a return value through it.
    default_light_ = selected_;
    if (const gpu::SceneNode* node = scene_graph_.find(default_light_); node != nullptr) {
        default_light_transform_ = node->local_transform;
    }
}

void Application::reposition_default_light() {
    const gpu::SceneNode* node = scene_graph_.find(default_light_);
    if (node == nullptr) {
        return;
    }
    // Only while it is still the light this put there. Once it has been moved
    // or aimed, it is the user's, and relocating it under them on the next
    // load would undo that.
    if (node->local_transform != default_light_transform_) {
        return;
    }

    const Bounds bounds = scene_bounds();
    if (bounds.empty()) {
        return;
    }

    const glm::vec3 centre = (bounds.min + bounds.max) * 0.5F;
    const float radius = std::max(glm::length(bounds.max - bounds.min) * 0.5F, 1e-3F);

    // Back along its own beam, which is where the light would be if it were a
    // sun -- so the cone icon points at the scene rather than away from it.
    const glm::vec3 direction =
        glm::normalize(glm::mat3(node->world_transform) * glm::vec3(0.0F, -1.0F, 0.0F));
    glm::vec3 position = centre - (direction * (radius * k_default_light_distance));

    // A nearly horizontal beam would leave the marker at the scene's own height
    // and back inside it, which is the thing this exists to avoid. The default
    // beam is not horizontal, but it is not the only one that reaches here.
    position.y = std::max(position.y, bounds.max.y + (radius * k_default_light_clearance));

    glm::mat4 transform = node->local_transform;
    // Only the translation column: the rotation is the aim, and rebuilding it
    // from the direction would round-trip through a basis for nothing.
    transform[3] = glm::vec4(position, 1.0F);

    scene_graph_.set_local_transform(default_light_, transform);
    scene_graph_.update_transforms();
    default_light_transform_ = transform;
}

// An addition is undone by tombstoning what it added and redone by putting it
// back. Neither touches the registries, so the geometry never moves -- which is
// the whole reason undo here is cheap.
void Application::push_add_command(std::string name, gpu::NodeHandle added,
                                   gpu::NodeHandle previous_selection) {
    push_command(
        std::move(name),
        [this, added, previous_selection]() {
            scene_graph_.remove_subtree(added);
            select(previous_selection);
        },
        [this, added]() {
            scene_graph_.restore_subtree(added);
            scene_graph_.update_transforms();
            select(added);
        });
}

void Application::commit_transform_edit() {
    if (!transform_edit_.has_value()) {
        return;
    }
    // Plain locals rather than a structured binding: a binding cannot be
    // captured by the lambdas below.
    const gpu::NodeHandle node = transform_edit_->first;
    const glm::mat4 before = transform_edit_->second;
    transform_edit_.reset();

    const gpu::SceneNode* current = scene_graph_.find(node);
    if (current == nullptr || current->local_transform == before) {
        // A click that moved nothing, or a node deleted mid-drag. Neither is an
        // edit, and an entry for it would make Ctrl+Z appear to do nothing.
        return;
    }
    const glm::mat4 after = current->local_transform;

    push_command(
        "Transform",
        [this, node, before]() {
            scene_graph_.set_local_transform(node, before);
            scene_graph_.update_transforms();
            select(node);
        },
        [this, node, after]() {
            scene_graph_.set_local_transform(node, after);
            scene_graph_.update_transforms();
            select(node);
        });
}

void Application::push_light_command(std::string name, gpu::NodeHandle node,
                                     const gpu::SceneLight& before, const gpu::SceneLight& after) {
    push_command(
        std::move(name),
        [this, node, before]() {
            scene_graph_.set_light(node, before);
            select(node);
        },
        [this, node, after]() {
            scene_graph_.set_light(node, after);
            select(node);
        });
}

void Application::commit_light_edit() {
    if (!light_edit_.has_value()) {
        return;
    }
    const gpu::NodeHandle node = light_edit_->first;
    const gpu::SceneLight before = light_edit_->second;
    light_edit_.reset();

    const gpu::SceneNode* current = scene_graph_.find(node);
    if (current == nullptr) {
        return;
    }
    push_light_command("Light", node, before, current->light);
}

void Application::push_command(std::string name, std::function<void()> undo_action,
                               std::function<void()> redo_action) {
    // Anything already undone is dropped: the history is a line, not a tree,
    // and a new edit made after stepping back replaces what was ahead.
    undo_stack_.resize(undo_cursor_);
    undo_stack_.push_back(Command{std::move(name), std::move(undo_action), std::move(redo_action)});

    // Bounded so a long session cannot grow it without limit. Dropping from the
    // front costs an O(n) shift on a vector, which happens once per edit past
    // the cap and is nothing next to the edit itself.
    if (undo_stack_.size() > k_max_undo_depth) {
        undo_stack_.erase(undo_stack_.begin());
    }
    undo_cursor_ = undo_stack_.size();
}

void Application::undo() {
    if (!can_undo()) {
        return;
    }
    --undo_cursor_;
    undo_stack_[undo_cursor_].undo();
    SAGE_LOG_INFO("Undo: {}", undo_stack_[undo_cursor_].name);
}

void Application::redo() {
    if (!can_redo()) {
        return;
    }
    undo_stack_[undo_cursor_].redo();
    SAGE_LOG_INFO("Redo: {}", undo_stack_[undo_cursor_].name);
    ++undo_cursor_;
}

void Application::clear_history() {
    undo_stack_.clear();
    undo_cursor_ = 0;
    transform_edit_.reset();
    light_edit_.reset();
}

void Application::delete_selected() {
    if (scene_graph_.find(selected_) == nullptr) {
        return;
    }
    const gpu::NodeHandle target = selected_;
    // Cleared first: the outline and the properties panel both read the
    // selection, and leaving it pointing at a tombstone would ask them to
    // describe something that is no longer there.
    select(gpu::NodeHandle{});
    scene_graph_.remove_subtree(target);

    // No wait_idle and no queueing, unlike a load or a clear. Nothing is freed
    // here -- the geometry stays exactly where it was -- so an in-flight
    // command buffer that still names it remains correct. All that changed is
    // which nodes the next frame walks.
    //
    // That is also why undoing this is just un-tombstoning: the mesh never
    // left, so there is nothing to upload again.
    push_command(
        "Delete",
        [this, target]() {
            scene_graph_.restore_subtree(target);
            scene_graph_.update_transforms();
            select(target);
        },
        [this, target]() {
            select(gpu::NodeHandle{});
            scene_graph_.remove_subtree(target);
        });

    SAGE_LOG_INFO("Deleted subtree; {} of {} slots live", scene_graph_.live_size(),
                  scene_graph_.size());
}

void Application::service_pending_light() {
    if (!pending_light_.has_value()) {
        return;
    }
    const PendingLight pending = *pending_light_;
    pending_light_.reset();

    if (!add_light(pending.type, pending.position)) {
        SAGE_LOG_ERROR("Could not add light: out of geometry capacity for its icon");
    }
}

void Application::service_pending_primitive() {
    if (!pending_primitive_.has_value()) {
        return;
    }
    const PendingPrimitive pending = *pending_primitive_;
    pending_primitive_.reset();

    if (!add_primitive(pending.kind, pending.position)) {
        SAGE_LOG_ERROR("Could not add {}: out of geometry capacity",
                       gpu::primitive_name(pending.kind));
    }
}

glm::vec3 Application::placement_point(float window_x, float window_y) const {
    const CameraMatrices matrices = camera_matrices();
    const glm::vec3 origin = camera_.position();

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const VkExtent2D extent = swapchain_.extent();
    if (viewport->Size.x <= 0.0F || viewport->Size.y <= 0.0F || viewport_rect_.extent.width == 0 ||
        viewport_rect_.extent.height == 0) {
        return origin + (camera_.forward() * k_fallback_placement_distance);
    }

    // ImGui reports logical coordinates; viewport_rect_ is framebuffer pixels.
    // Same conversion as central_node_rect, for the same reason.
    const float pixel_x = window_x * (static_cast<float>(extent.width) / viewport->Size.x);
    const float pixel_y = window_y * (static_cast<float>(extent.height) / viewport->Size.y);

    // Normalised within the 3D view, then to NDC. No Y negation: the
    // projection is already Vulkan's, whose NDC Y runs down the screen exactly
    // as framebuffer rows do.
    const float ndc_x = (2.0F * (pixel_x - static_cast<float>(viewport_rect_.offset.x)) /
                         static_cast<float>(viewport_rect_.extent.width)) -
                        1.0F;
    const float ndc_y = (2.0F * (pixel_y - static_cast<float>(viewport_rect_.offset.y)) /
                         static_cast<float>(viewport_rect_.extent.height)) -
                        1.0F;

    // Unprojecting the far plane alone is enough for a direction: the near
    // point is the camera position, which is already known exactly.
    const glm::mat4 inverse_view_projection = glm::inverse(matrices.projection * matrices.view);
    const glm::vec4 far_point = inverse_view_projection * glm::vec4(ndc_x, ndc_y, 1.0F, 1.0F);
    if (std::abs(far_point.w) < 1e-6F) {
        return origin + (camera_.forward() * k_fallback_placement_distance);
    }
    const glm::vec3 direction = glm::normalize(glm::vec3(far_point) / far_point.w - origin);

    // Intersect the ground plane. A ray running along it, or pointing away
    // from it, has no useful answer -- which is most of the time when the
    // camera is below the horizon -- so fall back to a fixed distance ahead.
    if (std::abs(direction.y) > 1e-4F) {
        const float distance = -origin.y / direction.y;
        if (distance > 0.0F && distance < k_max_placement_distance) {
            return origin + (direction * distance);
        }
    }
    return origin + (direction * k_fallback_placement_distance);
}

void Application::draw_context_menu() {
    if (ImGui::BeginPopup("viewport_context")) {
        ImGui::TextDisabled("Add at %.2f, %.2f, %.2f", static_cast<double>(context_menu_point_.x),
                            static_cast<double>(context_menu_point_.y),
                            static_cast<double>(context_menu_point_.z));
        ImGui::Separator();
        if (ImGui::MenuItem("Mesh (glTF)...")) {
            // Deferred: a popup cannot be opened from inside one that is about
            // to close, so this only records the intent. draw_file_dialog,
            // which runs outside any menu, opens it.
            file_dialog_open_ = true;
            // No replace or clear from here, and the load lands at the click.
            file_dialog_scene_actions_ = false;
            file_dialog_placement_ = context_menu_point_;
        }
        if (ImGui::BeginMenu("Primitive")) {
            constexpr std::array<gpu::PrimitiveKind, 5> k_kinds{
                gpu::PrimitiveKind::plane, gpu::PrimitiveKind::cube, gpu::PrimitiveKind::sphere,
                gpu::PrimitiveKind::cone, gpu::PrimitiveKind::cylinder};
            for (const gpu::PrimitiveKind kind : k_kinds) {
                if (ImGui::MenuItem(gpu::primitive_name(kind))) {
                    // Queued, not built here: the upload blocks on a transfer
                    // submission, and this is the middle of a frame.
                    pending_primitive_ = PendingPrimitive{kind, context_menu_point_};
                }
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Light")) {
            if (ImGui::MenuItem("Directional")) {
                pending_light_ = PendingLight{gpu::LightType::directional, context_menu_point_};
            }
            if (ImGui::MenuItem("Point")) {
                pending_light_ = PendingLight{gpu::LightType::point, context_menu_point_};
            }
            ImGui::EndMenu();
        }
        ImGui::EndPopup();
    }
}

void Application::draw_lighting_menu() {
    // Lights live in the scene graph now, so this panel holds only what is
    // global. Editing one light happens in Properties, with that light
    // selected -- the same place every other node is edited.
    std::uint32_t directional = 0;
    std::uint32_t point = 0;
    for (const gpu::SceneNode& node : scene_graph_.nodes()) {
        if (!node.alive || !node.has_light) {
            continue;
        }
        (node.light.type == gpu::LightType::directional ? directional : point) += 1;
    }
    ImGui::SeparatorText("Scene lights");
    ImGui::Text("%u directional, %u point", directional, point);
    if (directional + point > gpu::k_max_lights) {
        ImGui::TextDisabled("Over the %u the shader reads; the rest are ignored.",
                            gpu::k_max_lights);
    }
    if (directional == 0) {
        ImGui::TextDisabled("No directional light: nothing casts a shadow.");
    }
    ImGui::TextDisabled("Right-click the viewport to add one.");

    ImGui::SeparatorText("Ambient");
    ImGui::DragFloat("Intensity##ambient", &ambient_intensity_, 0.002F, 0.0F, 1.0F, "%.3f");
    ImGui::TextDisabled("Flat term; no IBL yet");

    ImGui::SeparatorText("Shadow");
    ImGui::Checkbox("Enabled##shadow", &shadows_enabled_);
    ImGui::BeginDisabled(!shadows_enabled_);
    // Acne and peter-panning are the two ends of one trade-off: too little bias
    // and the surface shadows itself in stripes, too much and the contact
    // shadow detaches from the object. The middle is found by dragging.
    // The constant term's range dwarfs the slope term's because the spec scales
    // it by the depth format's smallest resolvable difference -- about 2^-23
    // here. Hundreds is the working range, not single digits.
    ImGui::DragFloat("Depth bias", &shadow_depth_bias_, 10.0F, 0.0F, 10000.0F, "%.0f");
    ImGui::DragFloat("Slope bias", &shadow_slope_bias_, 0.05F, 0.0F, 8.0F, "%.2f");
    ImGui::DragFloat("Normal bias", &shadow_normal_bias_texels_, 0.05F, 0.0F, 8.0F, "%.2f texels");
    ImGui::SliderInt("PCF radius", &shadow_pcf_radius_, 0, 4);
    ImGui::EndDisabled();
    const int taps = ((2 * shadow_pcf_radius_) + 1) * ((2 * shadow_pcf_radius_) + 1);
    ImGui::TextDisabled("%ux%u map, %d taps", shadow_map_.resolution(), shadow_map_.resolution(),
                        taps);
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
    bool activated = false;
    bool released = false;

    edited |= ImGui::DragFloat3("Position", glm::value_ptr(translation), 0.01F);
    activated |= ImGui::IsItemActivated();
    released |= ImGui::IsItemDeactivatedAfterEdit();
    edited |= ImGui::DragFloat3("Rotation", glm::value_ptr(rotation), 0.5F);
    activated |= ImGui::IsItemActivated();
    released |= ImGui::IsItemDeactivatedAfterEdit();
    edited |= ImGui::DragFloat3("Scale", glm::value_ptr(scale), 0.01F);
    activated |= ImGui::IsItemActivated();
    released |= ImGui::IsItemDeactivatedAfterEdit();

    // Recorded before the edit is written below, so the stored value is the one
    // the drag began from.
    if (activated && !transform_edit_.has_value()) {
        transform_edit_ = std::make_pair(selected_, node->local_transform);
    }

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
    if (released) {
        commit_transform_edit();
    }

    ImGui::Separator();
    if (node->has_light) {
        ImGui::SeparatorText("Light");
        gpu::SceneLight light = node->light;

        int type = static_cast<int>(light.type);
        const bool type_changed = ImGui::Combo("Type", &type, "Directional\0Point\0");
        bool light_edited = type_changed;
        light.type = static_cast<gpu::LightType>(type);

        bool light_activated = false;
        bool light_released = false;
        light_edited |= ImGui::ColorEdit3("Colour", glm::value_ptr(light.color));
        light_activated |= ImGui::IsItemActivated();
        light_released |= ImGui::IsItemDeactivatedAfterEdit();
        light_edited |= ImGui::DragFloat("Intensity", &light.intensity, 0.1F, 0.0F, 500.0F);
        light_activated |= ImGui::IsItemActivated();
        light_released |= ImGui::IsItemDeactivatedAfterEdit();
        if (light.type == gpu::LightType::point) {
            light_edited |= ImGui::DragFloat("Range", &light.range, 0.05F, 0.01F, 500.0F);
            light_activated |= ImGui::IsItemActivated();
            light_released |= ImGui::IsItemDeactivatedAfterEdit();
        } else {
            // A directional light has no position, only a bearing, and the
            // rotate gizmo is how that is set.
            ImGui::TextDisabled("Rotate to aim; position is only the icon's.");
        }
        // The type combo commits in one go, so it is its own command rather
        // than the start of a drag.
        if (type_changed) {
            push_light_command("Change light type", selected_, node->light, light);
        }
        if (light_activated && !light_edit_.has_value()) {
            light_edit_ = std::make_pair(selected_, node->light);
        }
        if (light_edited) {
            scene_graph_.set_light(selected_, light);
        }
        if (light_released) {
            commit_light_edit();
        }
        ImGui::Separator();
    }

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

    // A drag runs over many frames and writes a transform on each. Recording
    // where it started and pushing once on release is what makes Ctrl+Z undo
    // the drag rather than one frame of it.
    if (ImGuizmo::IsUsing() && !transform_edit_.has_value()) {
        transform_edit_ = std::make_pair(selected_, node->local_transform);
    }
    if (!ImGuizmo::IsUsing()) {
        commit_transform_edit();
    }

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

    // Inside Begin, because GetStateStorage() returns the *current* window's,
    // and this one holds every tree node's open flag keyed by node index.
    if (hierarchy_state_stale_) {
        ImGui::GetStateStorage()->Clear();
        hierarchy_state_stale_ = false;
    }

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
        if (!nodes[i].alive) {
            continue;
        }
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

    // No DefaultOpen: a loaded file arrives collapsed to a single row named
    // after it, and is expanded on demand. A chess set is 50 nodes and a real
    // scene is more, which is a wall of names rather than an overview.
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
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
    ldr_target_.recreate(swapchain_.extent());
    // The old views are gone, so the descriptors naming them have to be
    // rewritten. The shadow map is absent here on purpose: it does not follow
    // the swapchain, so its view is still valid.
    bindless_set_.write_object_id_image(id_buffer_.view());
    bindless_set_.write_hdr_color_image(hdr_target_.view());
    bindless_set_.write_ldr_color_image(ldr_target_.view(), ldr_target_.sampler());
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
        service_pending_primitive();
        service_pending_light();
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

        // Before draw_dockspace, which reads it. Toggling after would submit a
        // dockspace this frame and hide the panels that belong in it, leaving
        // the 3D view sized for a layout that is no longer on screen.
        //
        // Not gated on wants_keyboard, unlike the gizmo keys: with the panels
        // hidden there is nothing left to take keyboard focus, so gating it
        // would make the toggle one-way in exactly the state it matters.
        if (ImGui::IsKeyPressed(ImGuiKey_F11)) {
            ui_hidden_ = !ui_hidden_;
        }

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
        // Gated on the keyboard, so Delete typed into the file dialog's path
        // field edits the text rather than the scene.
        if (!gpu::ImGuiLayer::wants_keyboard() && ImGui::IsKeyPressed(ImGuiKey_Delete)) {
            delete_selected();
        }
        if (!gpu::ImGuiLayer::wants_keyboard() && ImGui::GetIO().KeyCtrl) {
            // Ctrl+Y and Ctrl+Shift+Z both redo. The first is what was asked
            // for; the second is what a hand trained on other editors reaches
            // for, and supporting one does not cost the other.
            if (ImGui::IsKeyPressed(ImGuiKey_Y) ||
                (ImGui::GetIO().KeyShift && ImGui::IsKeyPressed(ImGuiKey_Z))) {
                redo();
            } else if (ImGui::IsKeyPressed(ImGuiKey_Z)) {
                undo();
            }
        }
        // Before picking: a drag that ends over a different object must not
        // also reselect it, and ImGuizmo only reports IsUsing() once drawn.
        const bool gizmo_active = draw_gizmo();

        // After draw_dockspace, which refreshes viewport_rect_, and before the
        // panels, so a click is tested against this frame's layout.
        if (!gizmo_active) {
            handle_picking_input();
        }

        // A right click that never became a camera drag. Gated the same way
        // picking is: a panel under the pointer wins, and a click outside the
        // 3D view is not a click in it.
        if (input.context_click && !gpu::ImGuiLayer::wants_mouse() &&
            viewport_texel(input.context_click_x, input.context_click_y).has_value()) {
            context_menu_point_ = placement_point(input.context_click_x, input.context_click_y);
            ImGui::OpenPopup("viewport_context");
        }
        // Outside the ui_hidden_ block: the menu is how things get added, and
        // hiding the panels for a capture should not take that away.
        draw_context_menu();

        if (!ui_hidden_) {
            draw_menu_bar();
            draw_stats_overlay();
            draw_hierarchy_panel();
            draw_properties_panel();
        }
        // Outside the block, like the context menu: the dialog is reachable
        // from both, and a menu that cannot open what it offers is worse than
        // no menu.
        draw_file_dialog();

        // Pass order is load-bearing. The shadow map is filled first, since the
        // scene samples it; the scene shades into the HDR target; the tonemap
        // resolves that to the LDR target; FXAA resolves *that* to the
        // swapchain and is the first thing to touch it; the capture takes the
        // finished image before any editor overlay reaches it; the outline and
        // then ImGui draw on top.
        write_frame_data(frame.slot);
        record_shadow(frame.command_buffer, frame.slot);
        record_scene(frame.command_buffer, frame.slot);
        record_tonemap(frame.command_buffer);
        record_fxaa(frame.command_buffer, acquired.image, swapchain_.image_view(acquired.index));
        if (!screenshot_include_ui_) {
            record_screenshot_copy(frame.command_buffer, acquired.image);
        }
        record_outline(frame.command_buffer, swapchain_.image_view(acquired.index));
        record_pick_copy(frame.command_buffer);
        imgui_.render(frame.command_buffer, swapchain_.image_view(acquired.index),
                      swapchain_.extent());
        // The other capture point: after everything, so the file shows the
        // editor rather than only what it is looking at.
        if (screenshot_include_ui_) {
            record_screenshot_copy(frame.command_buffer, acquired.image);
        }
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
