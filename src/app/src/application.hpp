#pragma once

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
#include <memory>
#include <optional>
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
    // Composes this frame's camera, lights and shadow matrix into the frame
    // buffer's slot. Separate from record_scene, and ahead of both passes,
    // because the shadow pass reads the same slot and needs the light matrix
    // that is in it.
    void write_frame_data(std::uint32_t frame_slot) const;
    // Draws the scene from the directional light into shadow_map_, depth only.
    // Runs before record_scene, which samples the result.
    void record_shadow(VkCommandBuffer command_buffer, std::uint32_t frame_slot) const;
    // Shades into hdr_target_, not the swapchain: values above 1.0 have to
    // survive as far as the tonemap, and an 8-bit attachment would clamp them
    // where they were written.
    void record_scene(VkCommandBuffer command_buffer, std::uint32_t frame_slot) const;
    // Resolves hdr_target_ to ldr_target_ through the tonemap curve, encoding
    // to sRGB on the way out because the destination is UNORM and does not do
    // it in fixed function.
    void record_tonemap(VkCommandBuffer command_buffer) const;
    // Anti-aliases ldr_target_ into the swapchain. Also where the swapchain
    // image first enters COLOR_ATTACHMENT_OPTIMAL, since nothing before this
    // point touches it.
    void record_fxaa(VkCommandBuffer command_buffer, VkImage image, VkImageView image_view) const;
    // Split out of record_scene because the UI draws into the same swapchain
    // image and must get there before it is handed to the presentation engine.
    static void transition_to_present(VkCommandBuffer command_buffer, VkImage image);
    // Lays a dockspace over the viewport and, on the first frame, docks the
    // panels into it. Panels place themselves by name from then on, which is
    // what stops a new one landing on top of an existing one.
    void draw_dockspace();
    void draw_ui();
    // Tonemap controls and the capture button. Grouped because both are about
    // how the frame is presented rather than what is in it.
    void draw_presentation_controls();
    // Lights and shadow tuning. A panel rather than constants because the
    // values that suit one model suit no other -- a point light placed for the
    // lantern is inside the board of a chess set -- and because shadow bias is
    // found by dragging a slider until acne stops without the contact shadow
    // detaching, which is not a thing to do one rebuild at a time.
    void draw_lighting_panel();
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

    // Turns a click in the 3D view into a pending object-id readback. No-op
    // when a panel has the pointer or the cursor is outside the viewport.
    void handle_picking_input();
    // W/E/R switch the manipulator, as in Unreal and Blender.
    void handle_gizmo_keys();
    // Copies the picked texel out of the id attachment. Recorded after the
    // scene, so the value read is the one this frame just drew.
    void record_pick_copy(VkCommandBuffer command_buffer) const;
    // Draws the selection outline over the scene, reading the id attachment
    // this frame just wrote. Runs before record_pick_copy, which is what fixes
    // each pass's expected source layout to a single value.
    void record_outline(VkCommandBuffer command_buffer, VkImageView image_view) const;
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

    // Loads a file into the registries and the graph. Returns false when the
    // file could not be read; the scene is left as it was in that case, unless
    // `replace` already emptied it.
    bool load_model(const std::filesystem::path& path, bool replace);
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

    struct Bounds {
        glm::vec3 min{0.0F};
        glm::vec3 max{0.0F};
        [[nodiscard]] bool empty() const { return min.x > max.x; }
    };
    // World-space AABB over every mesh currently in the graph, recomputed from
    // live world transforms rather than remembered from load time -- otherwise
    // moving a node with the gizmo would leave the shadow frustum behind.
    [[nodiscard]] Bounds scene_bounds() const;

    struct LightFit {
        glm::mat4 view_projection{1.0F};
        // How much world space one shadow-map texel covers. The unit the
        // normal-offset bias is expressed in, so that it means the same thing
        // whatever the scene's scale.
        float world_texel_size = 0.0F;
    };
    // An orthographic light-space matrix fitted to `bounds`. A directional
    // light has no position, so the frustum is placed by the scene rather than
    // by the light: it is centred on the bounds and pulled back far enough
    // along the light direction to enclose them.
    [[nodiscard]] LightFit fit_light(const Bounds& bounds) const;

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

    // Scene lighting, editable rather than hardcoded. Two lights: a key
    // directional one, which is the only one that casts a shadow, and an
    // optional point light for fill.
    struct DirectionalLightState {
        // Pointing direction, i.e. the way the light travels. Normalised before
        // upload; the UI edits it as a raw vector because a pair of angles is
        // harder to reason about when matching a reference image.
        // Angled rather than near-vertical. The old default was mostly
        // straight down, which puts every shadow directly underneath the thing
        // casting it -- correct, and invisible from any normal camera.
        glm::vec3 direction{-0.6F, -0.55F, -0.6F};
        glm::vec3 color{1.0F, 0.96F, 0.9F};
        // 2.0 was the M5 value and leaves a typical glTF sitting around a
        // quarter of the display range, where a shadow has no room to be
        // darker than its surroundings. Measured on the chess set: at 2 the
        // subject averages 45/255, at 25 it averages 124 with shadows a clear
        // 98 levels below. This is a middle that suits most files; the slider
        // is there for the ones it does not.
        float intensity = 8.0F;
    };
    struct PointLightState {
        glm::vec3 position{0.0F, 1.0F, 0.0F};
        glm::vec3 color{1.0F, 0.7F, 0.35F};
        float intensity = 4.0F;
        float range = 4.0F;
        bool enabled = false;
    };
    DirectionalLightState key_light_;
    PointLightState fill_light_;
    // Replaces the 0.03 that was compiled into the shader. Without ambient
    // occlusion or IBL this is the only thing keeping unlit faces off pure
    // black, so it is the difference between "dramatic" and "half the model is
    // missing" -- which is a judgement call, hence a slider.
    float ambient_intensity_ = 0.03F;

    bool shadows_enabled_ = true;
    // Hardware depth bias, applied while rasterising the shadow map. The spec's
    // offset is `m * slopeFactor + r * constantFactor`, and the two halves are
    // in wildly different units: m is the depth slope, but r is the smallest
    // resolvable depth difference, which for a D32_SFLOAT map is around 2^-23.
    // So a slope factor of 2 is meaningful while a constant factor of 2 is
    // worth about 1e-7 -- nothing. Measured on this hardware: 1.5 was
    // indistinguishable from 0, and visible change started in the hundreds.
    // Hence the scale difference between these two defaults.
    float shadow_depth_bias_ = 500.0F;
    float shadow_slope_bias_ = 2.0F;
    // Applied at lookup time instead, along the surface normal, and measured in
    // shadow-map texels rather than world units. Texels because the frustum is
    // refitted to the scene every frame: a bias of "0.02 world units" is
    // nothing on a cathedral and four percent of a chess set, so an absolute
    // value cannot have one sensible default. A texel is the unit the error
    // actually scales with.
    float shadow_normal_bias_texels_ = 1.5F;
    int shadow_pcf_radius_ = 2;

    // Which curve the tonemap applies. Stored as int to match the push
    // constant; the values are TonemapOperator in application.cpp, which must
    // agree with shaders/tonemap.slang.
    int tonemap_operator_ = 0;
    // Exposure in stops, which is the unit it is reasoned about in. Converted
    // to the linear multiplier the shader wants at push time.
    float exposure_stops_ = 0.0F;

    // The FXAA pass runs unconditionally and passes the image through when this
    // is off, rather than being skipped. Skipping it would mean the tonemap
    // writing to a different target depending on a checkbox, and two barrier
    // paths to keep correct; a branch in the shader costs a comparison.
    bool fxaa_enabled_ = true;
    float fxaa_edge_threshold_ = 0.125F;
    float fxaa_edge_threshold_min_ = 0.0312F;
    float fxaa_subpixel_quality_ = 0.75F;

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
    std::optional<FilePicker::Request> pending_load_;
    bool pending_clear_ = false;

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
    gpu::Buffer frame_buffer_;
    // One texel of object id, copied out of id_buffer_ on a pick. Host-cached
    // rather than write-combined: this one is read, not written.
    gpu::Buffer pick_buffer_;
    gpu::SceneGraph scene_graph_;
    gpu::Swapchain swapchain_;
    gpu::DepthBuffer depth_buffer_;
    gpu::IdBuffer id_buffer_;
    gpu::HdrTarget hdr_target_;
    gpu::LdrTarget ldr_target_;
    // Not recreated on resize: its resolution is a quality setting, not a
    // consequence of the window.
    gpu::ShadowMap shadow_map_;
    gpu::PipelineCache pipeline_cache_;
    gpu::GraphicsPipeline shadow_pipeline_;
    gpu::GraphicsPipeline pipeline_;
    gpu::GraphicsPipeline outline_pipeline_;
    gpu::GraphicsPipeline tonemap_pipeline_;
    gpu::GraphicsPipeline fxaa_pipeline_;
    gpu::FramePacer frame_pacer_;
    // Last, so it is destroyed first: its teardown frees Vulkan objects and
    // touches the device, both of which must still be alive.
    gpu::ImGuiLayer imgui_;
};

}  // namespace sage::app
