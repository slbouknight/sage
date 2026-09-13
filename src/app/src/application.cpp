#include "application.hpp"

#include <sage/core/log.hpp>
#include <sage/core/math.hpp>
#include <sage/gpu/geometry_registry.hpp>
#include <sage/gpu/light.hpp>
#include <sage/gpu/screenshot.hpp>
#include <sage/gpu/selection_buffer.hpp>
#include <sage/gpu/vertex.hpp>
#include <sage/gpu/vk_check.hpp>

#include <imgui.h>

// After imgui.h: it uses ImGui's types without including it.
#include <ImGuizmo.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <memory>
#include <numbers>
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

// Materials are 64 bytes, so this is 64 KiB of device memory for the whole
// table. Cheap enough that overrunning it is a reason to raise the number
// rather than a case MaterialRegistry has to degrade around.
constexpr std::uint32_t k_max_materials = 1024;

// Captures land here, relative to the working directory. Git-ignored: these are
// output, and a portfolio screenshot belongs in a README by hand rather than
// accumulating in the tree.
constexpr const char* k_screenshot_directory = "screenshots";

// Bytes per pixel in the swapchain format, for sizing the readback buffer.
// Every format is_capturable_format accepts is 8-bit RGBA or BGRA.
constexpr VkDeviceSize k_swapchain_bytes_per_pixel = 4;

// Used by the capture copy below. The render passes have their own copy in
// renderer.cpp; a shared header for two lines of constexpr would cost more
// than it saved.
constexpr VkImageSubresourceRange k_color_range{
    VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};

// 720p left the 3D view around 716 px wide once the panels took their columns,
// which is thin for an editor and thinner still for a capture of one.
constexpr std::uint32_t k_initial_width = 1600;
constexpr std::uint32_t k_initial_height = 900;

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
      swapchain_(device_, surface_.handle(), window_.framebuffer_extent()),
      renderer_(device_, allocator_, bindless_set_, swapchain_),
      frame_pacer_(device_),
      imgui_(instance_, device_, window_, swapchain_.format(), swapchain_.image_count()) {
    gizmo_operation_ = static_cast<int>(ImGuizmo::TRANSLATE);
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

// Gathers everything the render passes read from outside themselves, once per
// frame. The single place the editor's state is handed to the renderer -- which
// is what keeps the passes from reaching back into Application for it.
Renderer::FrameView Application::frame_view(std::uint32_t frame_slot) const {
    const CameraMatrices matrices = camera_matrices();

    // The shadow map is fitted to one directional light -- the first in the
    // graph, matching the shader, which spends its one lookup on the first
    // directional light it evaluates. With none in the scene there is nothing
    // to fit and nothing to cast.
    const gpu::SceneNode* key = scene_graph_.find(first_directional_light());

    Renderer::FrameView view;
    view.scene = &scene_graph_;
    view.geometry = &geometry_registry_;
    view.settings = &render_settings_;
    view.view_projection = matrices.projection * matrices.view;
    view.camera_position = camera_.position();
    view.has_key_light = key != nullptr;
    if (key != nullptr) {
        view.key_light_direction =
            glm::vec3(glm::mat3(key->world_transform) * glm::vec3(0.0F, -1.0F, 0.0F));
    }
    view.viewport = viewport_rect_;
    // Hidden with the panels, and hidden for the frame a no-UI capture is
    // taken on. That frame is also the one on screen, so pressing F2 without
    // F11 blinks the icons for a single frame -- the alternative is rendering
    // the scene twice.
    view.show_editor_meshes =
        !ui_hidden_ && !(pending_screenshot_.has_value() && !screenshot_include_ui_);
    view.frame_slot = frame_slot;
    return view;
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

gpu::NodeHandle Application::first_directional_light() const {
    for (std::size_t i = 0; i < scene_graph_.nodes().size(); ++i) {
        const gpu::SceneNode& node = scene_graph_.nodes()[i];
        if (node.alive && node.has_light && node.light.type == gpu::LightType::directional) {
            return scene_graph_.handle_at(i);
        }
    }
    return gpu::NodeHandle{};
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
    return view_mapping().texel_at(glm::vec2{window_x, window_y});
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

    const std::uint32_t id = renderer_.read_picked_id();

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
    const Bounds bounds = compute_scene_bounds(scene_graph_);
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

    const Bounds bounds = compute_scene_bounds(scene_graph_);
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

    const Bounds bounds = compute_scene_bounds(scene_graph_);
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
    history_.push(
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

    history_.push(
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
    history_.push(
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

void Application::clear_history() {
    history_.clear();
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
    history_.push(
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
    const glm::vec3 origin = camera_.position();
    // Where the ray misses, or the view is not yet laid out: a fixed distance
    // straight ahead, which is at least visible and in front of the camera.
    const glm::vec3 fallback = origin + (camera_.forward() * k_fallback_placement_distance);

    const std::optional<glm::vec2> ndc = view_mapping().ndc_at(glm::vec2{window_x, window_y});
    if (!ndc.has_value()) {
        return fallback;
    }

    const CameraMatrices matrices = camera_matrices();
    const std::optional<glm::vec3> direction =
        ray_direction(glm::inverse(matrices.projection * matrices.view), origin, *ndc);
    if (!direction.has_value()) {
        return fallback;
    }

    const std::optional<glm::vec3> hit =
        ground_plane_hit(origin, *direction, k_max_placement_distance);
    return hit.value_or(origin + (*direction * k_fallback_placement_distance));
}

// The 3D view's placement, gathered from ImGui and the swapchain. Built per
// call rather than cached: the ImGui read is trivial, and a cached copy would
// be one more thing to invalidate on resize.
ViewportMapping Application::view_mapping() const {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ViewportMapping mapping;
    mapping.logical_pos = glm::vec2{viewport->Pos.x, viewport->Pos.y};
    mapping.logical_size = glm::vec2{viewport->Size.x, viewport->Size.y};
    mapping.framebuffer = swapchain_.extent();
    mapping.rect = viewport_rect_;
    return mapping;
}

bool Application::recreate_swapchain() {
    VkExtent2D extent = window_.framebuffer_extent();
    if (extent.width == 0 || extent.height == 0) {
        return false;
    }
    swapchain_.recreate(extent);
    renderer_.recreate(swapchain_.extent());
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
                history_.redo();
            } else if (ImGui::IsKeyPressed(ImGuiKey_Z)) {
                history_.undo();
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
        const Renderer::FrameView view = frame_view(frame.slot);
        renderer_.write_frame_data(view);
        renderer_.record_shadow(frame.command_buffer, view);
        renderer_.record_scene(frame.command_buffer, view);
        renderer_.record_tonemap(frame.command_buffer, view);
        renderer_.record_fxaa(frame.command_buffer, acquired.image,
                              swapchain_.image_view(acquired.index), view);
        if (!screenshot_include_ui_) {
            record_screenshot_copy(frame.command_buffer, acquired.image);
        }
        renderer_.record_outline(frame.command_buffer, swapchain_.image_view(acquired.index), view,
                                 selected_.valid());
        if (pending_pick_.has_value()) {
            renderer_.record_pick_copy(frame.command_buffer, *pending_pick_);
        }
        imgui_.render(frame.command_buffer, swapchain_.image_view(acquired.index),
                      swapchain_.extent());
        // The other capture point: after everything, so the file shows the
        // editor rather than only what it is looking at.
        if (screenshot_include_ui_) {
            record_screenshot_copy(frame.command_buffer, acquired.image);
        }
        Renderer::transition_to_present(frame.command_buffer, acquired.image);

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
    renderer_.save_pipeline_cache();
    SAGE_LOG_INFO("Main loop exited");
}

}  // namespace sage::app
