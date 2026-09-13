#pragma once

#include <sage/app/edit_history.hpp>
#include <sage/app/render_settings.hpp>
#include <sage/app/renderer.hpp>
#include <sage/app/scene_query.hpp>
#include <sage/app/viewport_mapping.hpp>
#include <sage/core/camera.hpp>
#include <sage/gpu/allocator.hpp>
#include <sage/gpu/bindless_set.hpp>
#include <sage/gpu/buffer.hpp>
#include <sage/gpu/depth_buffer.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/frame_pacer.hpp>
#include <sage/gpu/geometry_registry.hpp>
#include <sage/gpu/gltf_loader.hpp>
#include <sage/gpu/hdr_target.hpp>
#include <sage/gpu/id_buffer.hpp>
#include <sage/gpu/imgui_layer.hpp>
#include <sage/gpu/instance.hpp>
#include <sage/gpu/ldr_target.hpp>
#include <sage/gpu/material_registry.hpp>
#include <sage/gpu/pipeline.hpp>
#include <sage/gpu/pipeline_cache.hpp>
#include <sage/gpu/primitives.hpp>
#include <sage/gpu/sampler.hpp>
#include <sage/gpu/scene.hpp>
#include <sage/gpu/selection_buffer.hpp>
#include <sage/gpu/shadow_map.hpp>
#include <sage/gpu/surface.hpp>
#include <sage/gpu/swapchain.hpp>
#include <sage/gpu/texture_registry.hpp>
#include <sage/gpu/uploader.hpp>
#include <sage/gpu/window.hpp>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "file_picker.hpp"

namespace sage::app {

class Application {
public:
    explicit Application(const std::filesystem::path& model_path);
    ~Application();

    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
    Application(Application&&) = delete;
    Application& operator=(Application&&) = delete;

    void run();

private:
    // Returns false when the window is minimized and the frame should be
    // skipped entirely.
    bool recreate_swapchain();
    // Lays a dockspace over the viewport and, on the first frame, docks the
    // panels into it. Panels place themselves by name from then on, which is
    // what stops a new one landing on top of an existing one.
    void draw_dockspace();
    // The application menu bar across the top. Holds everything global -- what
    // to load, how the frame is presented, how the scene is lit -- so those
    // stop occupying a docked panel each and the 3D view gets the width back.
    // Keeps the logical-coordinate copy of the 3D view's rect in step with the
    // pixel one. Takes the dockspace id as unsigned int rather than ImGuiID to
    // keep imgui out of this header.
    void update_viewport_logical_rect(unsigned int dockspace);
    void draw_menu_bar();
    // The read-out, floating over the top-right of the 3D view rather than
    // docked. It is a heads-up display: always wanted, never interacted with,
    // and a panel's worth of screen is too much to pay for it.
    void draw_stats_overlay();
    // The modal glTF browser. One dialog, two callers: the File menu opens it
    // over the whole scene, the context menu opens it to add at a point.
    void draw_file_dialog();
    // Tonemap, anti-aliasing and capture. Drawn as menu content rather than a
    // panel, so it lives wherever it is opened from.
    void draw_presentation_controls();
    // Lights and shadow tuning. A panel rather than constants because the
    // values that suit one model suit no other -- a point light placed for the
    // lantern is inside the board of a chess set -- and because shadow bias is
    // found by dragging a slider until acne stops without the contact shadow
    // detaching, which is not a thing to do one rebuild at a time.
    void draw_lighting_menu();
    void draw_hierarchy_panel();
    void draw_properties_panel();
    // Draws the manipulator and writes any drag back into the scene graph.
    // Returns true while the gizmo is being dragged, which suppresses picking
    // so that releasing over another object does not reselect.
    bool draw_gizmo();

    struct CameraMatrices {
        glm::mat4 view{1.0F};
        // Vulkan convention, with Y flipped for NDC. What the scene renders with.
        glm::mat4 projection{1.0F};
        // The same projection without the flip, for ImGuizmo.
        glm::mat4 projection_gl{1.0F};
    };
    // Derived from the camera and the current viewport rect, so the gizmo and
    // the scene pass cannot disagree about where a point lands on screen.
    [[nodiscard]] CameraMatrices camera_matrices() const;
    // Gathers everything the render passes read from outside themselves. The
    // single place the editor's state crosses into the renderer.
    [[nodiscard]] Renderer::FrameView frame_view(std::uint32_t frame_slot) const;

    // Turns a click in the 3D view into a pending object-id readback. No-op
    // when a panel has the pointer or the cursor is outside the viewport.
    void handle_picking_input();
    // A window position in ImGui's logical coordinates as a framebuffer texel,
    // or nothing when it falls outside the 3D view. Shared by picking and by
    // the context menu, which need the identical test.
    [[nodiscard]] std::optional<VkOffset2D> viewport_texel(float window_x, float window_y) const;
    // The 3D view's placement, read from ImGui and the swapchain. The one
    // place that reaches for ImGui's viewport; everything downstream works on
    // the plain values.
    [[nodiscard]] ViewportMapping view_mapping() const;
    // W/E/R switch the manipulator, as in Unreal and Blender.
    void handle_gizmo_keys();
    // Reads the copied texel back and resolves it to a node. Must run only
    // after the submission carrying record_pick_copy has completed.
    void resolve_pick();

    // Queues a capture for the end of this frame. Names the file here rather
    // than at write time so a burst of captures cannot collide on a timestamp.
    void request_screenshot();
    // Copies the viewport out of the tonemapped swapchain image. Recorded
    // between the tonemap and the outline, so the file holds the rendered image
    // and none of the editor's overlays. Non-const: it allocates the readback
    // buffer, whose size is not known until the viewport rect is.
    void record_screenshot_copy(VkCommandBuffer command_buffer, VkImage image);
    // Encodes the copied pixels to PNG and releases the readback buffer. Same
    // rule as resolve_pick: only after the submission carrying the copy is done.
    void resolve_screenshot();
    // Records a new selection. The flag upload it implies is deferred rather
    // than done here, because this is reachable from inside a panel's draw.
    void select(gpu::NodeHandle node);
    // Uploads the selection flags when they have changed. Runs at the top of a
    // frame, before anything is recorded, for the same reason a queued load
    // does: the upload blocks and submits work of its own.
    void service_selection();

    // A queued load. Separate from FilePicker::Request because the context menu
    // raises these too, and it places what it adds where the click landed.
    struct PendingLoad {
        std::filesystem::path path;
        bool replace = true;
        // Absent means "wherever the file says", which is what the panel and
        // the command line want. Set, it moves the load's root node there.
        std::optional<glm::vec3> placement;
    };

    // Loads a file into the registries and the graph. Returns false when the
    // file could not be read; the scene is left as it was in that case, unless
    // `replace` already emptied it.
    bool load_model(const PendingLoad& load);

    // Where a right click in the viewport points, as a world position: the
    // cursor ray intersected with the ground plane. Objects added from the
    // context menu land here.
    [[nodiscard]] glm::vec3 placement_point(float window_x, float window_y) const;

    struct PendingPrimitive {
        gpu::PrimitiveKind kind = gpu::PrimitiveKind::plane;
        glm::vec3 position{0.0F};
    };
    // Queued for the same reason a load is: generating one is cheap, but
    // uploading it blocks on a transfer submission, which is not a thing to do
    // with a command buffer already recording.
    std::optional<PendingPrimitive> pending_primitive_;
    void service_pending_primitive();

    struct PendingLight {
        gpu::LightType type = gpu::LightType::directional;
        glm::vec3 position{0.0F};
    };
    // Queued for the same reason: adding one uploads its icon mesh.
    std::optional<PendingLight> pending_light_;
    void service_pending_light();
    // Deletes the selection and everything beneath it. Immediate rather than
    // queued: nothing is freed, so no in-flight command buffer is invalidated.
    void delete_selected();

    // Records an addition, which undo tombstones and redo restores. The
    // closures capture by value and `this`, which outlives them.
    void push_add_command(std::string name, gpu::NodeHandle added,
                          gpu::NodeHandle previous_selection);
    // Closes a transform drag, pushing one command for the whole gesture.
    // Does nothing when the value came back unchanged.
    void commit_transform_edit();
    void commit_light_edit();
    void push_light_command(std::string name, gpu::NodeHandle node, const gpu::SceneLight& before,
                            const gpu::SceneLight& after);
    // Drops the history and any edit mid-gesture. Called when the registries
    // rewind, which is the one thing here that genuinely cannot be reversed.
    void clear_history();

    EditHistory history_;

    // A transform edit in progress, and the value it started from. Both the
    // gizmo and the Properties drags run across many frames, so the command is
    // pushed on release -- otherwise one drag would leave a hundred entries and
    // Ctrl+Z would rewind a frame at a time.
    std::optional<std::pair<gpu::NodeHandle, glm::mat4>> transform_edit_;
    // The same, for the light fields in the Properties panel.
    std::optional<std::pair<gpu::NodeHandle, gpu::SceneLight>> light_edit_;
    // Builds a primitive, sizes it against the scene, and drops it in. Returns
    // false when the geometry did not fit.
    bool add_primitive(gpu::PrimitiveKind kind, const glm::vec3& position);
    // The right-click menu over the 3D view, and the modal browser it can open.
    void draw_context_menu();
    // Rewinds all three registries and empties the graph. Waits for the device
    // to go idle first: in-flight command buffers still name this geometry, and
    // TextureRegistry::reset destroys live images.
    void clear_scene();
    // Runs a queued load at the top of a frame, before any recording. Loading
    // from inside the picker's own draw call would mean blocking uploads and a
    // wait_idle in the middle of a frame whose command buffer is already begun.
    void service_pending_load();
    // Children per node, indexed by node index. Rebuilt each frame rather than
    // stored; see draw_hierarchy_panel for why.
    using ChildTable = std::vector<std::vector<std::uint32_t>>;
    void draw_hierarchy_node(std::uint32_t index, const ChildTable& children);
    void frame_camera_on(const glm::vec3& bounds_min, const glm::vec3& bounds_max);

    // Bounds, LightFit and the functions over them live in scene_query.hpp:
    // they are arithmetic over the graph, and were only members because
    // everything here was.

    // The light the shadow map is fitted to: the first directional one in the
    // graph, which is also the one the shader shadows. Invalid when there is
    // none, in which case nothing casts.
    [[nodiscard]] gpu::NodeHandle first_directional_light() const;
    // Creates the default key light. Called at startup and after a clear, since
    // a light is a node and clearing the graph removes it -- without this, an
    // empty scene would load the next model into the dark.
    void add_default_light();
    // Moves the default light to suit the scene, once there is one to measure.
    // Does nothing once the light has been touched: from that point it is the
    // user's, not a default.
    void reposition_default_light();
    // Adds a light node, with the small mesh that makes it visible and
    // clickable, at `position`.
    // `record` false for the default light, which is setup rather than an edit
    // and must not be the first thing Ctrl+Z reaches for.
    bool add_light(gpu::LightType type, const glm::vec3& position, bool record = true);

    // The light add_default_light created, and the transform it was left with.
    // Comparing against that transform is how "still a default" is decided --
    // cheaper and more honest than a dirty flag, which every edit path would
    // have to remember to set.
    gpu::NodeHandle default_light_;
    glm::mat4 default_light_transform_{1.0F};

    bool dock_layout_built_ = false;
    // Set when node indices are about to be reused, so the hierarchy panel
    // drops ImGui's remembered tree open/closed state on its next draw.
    bool hierarchy_state_stale_ = false;

    // The 3D view's rect within the swapchain image: the dockspace's central
    // node, in framebuffer pixels. The scene is drawn here rather than across
    // the whole image, so it is neither hidden behind panels nor framed for a
    // viewport wider than the visible one.
    VkRect2D viewport_rect_{};

    struct SceneBounds {
        glm::vec3 min{0.0F};
        glm::vec3 max{0.0F};
    };
    // Framing is deferred to just after the dockspace is laid out, because it
    // depends on the central node's aspect ratio -- which does not exist yet
    // when the constructor loads a model named on the command line.
    std::optional<SceneBounds> pending_frame_;

    // Texel in the id attachment to read back, in framebuffer pixels. Set on
    // click and cleared once resolved.
    std::optional<VkOffset2D> pending_pick_;
    gpu::NodeHandle selected_;
    // One flag per node: 1 for the selection and everything beneath it. Kept
    // here rather than rebuilt per frame so the upload can be skipped when
    // nothing changed.
    std::vector<std::uint32_t> selection_flags_;
    bool selection_dirty_ = false;
    // Which manipulator is active. Stored as int to keep ImGuizmo's enum out
    // of this header; application.cpp casts it back.
    // ImGuizmo::TRANSLATE, assigned in the constructor body so the enum stays
    // out of this header.
    int gizmo_operation_ = 0;
    bool gizmo_local_space_ = false;

    // Everything a person can turn about how the frame is shaded and
    // presented. One struct rather than fifteen members because it is the
    // seam: the panels write it, the render passes read it.
    RenderSettings render_settings_;

    // Hides every panel and gives the 3D view the whole window. Two things need
    // it: a capture at full window resolution rather than whatever the central
    // dock node happens to be, and simply looking at the render.
    bool ui_hidden_ = false;
    // Whether a capture includes the editor around the render. Off is the
    // portfolio frame; on is the one that shows the editor is an editor, which
    // a README wants and the clean frame cannot show.
    bool screenshot_include_ui_ = false;

    // Where the next capture goes, set on request and cleared once written.
    std::optional<std::filesystem::path> pending_screenshot_;
    // Allocated only while a capture is in flight. A permanent readback buffer
    // would cost a swapchain's worth of host memory -- 33 MiB at 4K -- for a
    // feature used a handful of times a session.
    std::unique_ptr<gpu::Buffer> screenshot_buffer_;
    // The region actually copied, kept because viewport_rect_ may have moved by
    // the time the pixels are read back.
    VkExtent2D screenshot_extent_{};

    FilePicker file_picker_;
    std::optional<PendingLoad> pending_load_;
    bool pending_clear_ = false;
    // Where the context menu was opened, in world space. Held for as long as
    // the menu and any modal it spawns are up, so what gets added lands where
    // the user clicked rather than where the camera happens to be by then.
    glm::vec3 context_menu_point_{0.0F};
    // The glTF browser's state. Opened from either menu, and outliving the one
    // that opened it -- a popup cannot be nested inside one that is closing.
    bool file_dialog_open_ = false;
    // Whether the dialog offers Replace and Clear. The File menu is operating
    // on the scene as a whole and wants them; the context menu's verb is
    // "add", where replacing the scene is not an answer to anything.
    bool file_dialog_scene_actions_ = false;
    // Where the load lands, when it was asked for at a point.
    std::optional<glm::vec3> file_dialog_placement_;

    // The 3D view again, in ImGui's logical coordinates rather than framebuffer
    // pixels. The overlay is placed with it, and ImGui positions windows in
    // logical units -- converting viewport_rect_ back every frame would be
    // undoing a conversion that was made three lines earlier.
    glm::vec2 viewport_logical_pos_{0.0F};
    glm::vec2 viewport_logical_size_{0.0F};

    core::Camera camera_;
    gpu::Window window_;
    gpu::Instance instance_;
    gpu::Surface surface_;
    gpu::PhysicalDeviceInfo physical_device_;
    gpu::Device device_;
    gpu::Allocator allocator_;
    gpu::BindlessSet bindless_set_;
    gpu::Uploader uploader_;
    gpu::GeometryRegistry geometry_registry_;
    gpu::Sampler sampler_;
    gpu::TextureRegistry texture_registry_;
    gpu::MaterialRegistry material_registry_;
    gpu::SelectionBuffer selection_buffer_;
    gpu::SceneGraph scene_graph_;
    gpu::Swapchain swapchain_;
    // Every target and pipeline, and the passes over them.
    Renderer renderer_;
    gpu::FramePacer frame_pacer_;
    // Last, so it is destroyed first: its teardown frees Vulkan objects and
    // touches the device, both of which must still be alive.
    gpu::ImGuiLayer imgui_;
};

}  // namespace sage::app
