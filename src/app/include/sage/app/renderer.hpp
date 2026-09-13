#pragma once

#include <sage/app/render_settings.hpp>
#include <sage/core/math.hpp>
#include <sage/gpu/allocator.hpp>
#include <sage/gpu/bindless_set.hpp>
#include <sage/gpu/buffer.hpp>
#include <sage/gpu/depth_buffer.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/geometry_registry.hpp>
#include <sage/gpu/hdr_target.hpp>
#include <sage/gpu/id_buffer.hpp>
#include <sage/gpu/ldr_target.hpp>
#include <sage/gpu/pipeline.hpp>
#include <sage/gpu/pipeline_cache.hpp>
#include <sage/gpu/scene.hpp>
#include <sage/gpu/shadow_map.hpp>
#include <sage/gpu/swapchain.hpp>

#include <cstdint>

namespace sage::app {

// Every render target and pipeline, and the passes over them.
//
// Split out of Application because it was the half that grows: the editor's
// panels are roughly done, while a deferred path or ray-traced shadows would
// each add targets and passes to what was already the largest file in the
// repo. It owns the resources those passes read and write; everything else --
// the scene, the camera, what the user has selected -- arrives per frame in a
// FrameView rather than being reached for.
//
// What it deliberately does not own: the swapchain (the window's, and it is
// presented by the frame loop), the registries (filled by loading, which is
// the editor's job), and capture (host readback and PNG writing, which is I/O
// rather than rendering).
class Renderer {
public:
    // `swapchain` and `bindless_set` are borrowed and must outlive this.
    Renderer(gpu::Device& device, gpu::Allocator& allocator, gpu::BindlessSet& bindless_set,
             const gpu::Swapchain& swapchain);

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(Renderer&&) = delete;
    ~Renderer() = default;

    // Everything outside the renderer that a pass reads, gathered once per
    // frame by the caller.
    //
    // A struct rather than a dozen parameters repeated across seven passes,
    // and pointers rather than references so it stays an aggregate. Nothing in
    // here is owned; it must all outlive the call.
    struct FrameView {
        const gpu::SceneGraph* scene = nullptr;
        const gpu::GeometryRegistry* geometry = nullptr;
        const RenderSettings* settings = nullptr;

        glm::mat4 view_projection{1.0F};
        glm::vec3 camera_position{0.0F};

        // The direction the shadow map is fitted to, from the scene's first
        // directional light -- matching the shader, which spends its one
        // shadow lookup on the first directional light it evaluates. With
        // has_key_light false nothing casts, though the pass still runs.
        glm::vec3 key_light_direction{0.0F, -1.0F, 0.0F};
        bool has_key_light = false;

        // The 3D view within the swapchain image. The scene is drawn here
        // rather than across the whole image, so it is neither hidden behind
        // panels nor framed for a viewport wider than the visible one.
        VkRect2D viewport{};

        // Whether to draw editor_only meshes -- the icons standing for lights.
        // They are drawn with the scene so they pick and outline like any
        // other mesh, but they are the tool rather than the render: hidden
        // with the panels, and hidden for the frame a no-UI capture is taken
        // on. Decided by the caller, which is the only thing that knows a
        // capture is pending.
        bool show_editor_meshes = true;

        // Which slot of the per-frame buffer this frame owns.
        std::uint32_t frame_slot = 0;
    };

    // Composes this frame's camera, lights and shadow matrix into the frame
    // buffer's slot. Separate from record_scene, and ahead of both passes,
    // because the shadow pass reads the same slot and needs the light matrix
    // that is in it.
    void write_frame_data(const FrameView& view) const;

    // Draws the scene from the directional light, depth only. Runs before
    // record_scene, which samples the result.
    void record_shadow(VkCommandBuffer command_buffer, const FrameView& view) const;

    // Shades into the HDR target, not the swapchain: values above 1.0 have to
    // survive as far as the tonemap, and an 8-bit attachment would clamp them
    // where they were written.
    void record_scene(VkCommandBuffer command_buffer, const FrameView& view) const;

    // Resolves the HDR target to the LDR one through the tonemap curve,
    // encoding to sRGB on the way out because the destination is UNORM and
    // does not do it in fixed function.
    void record_tonemap(VkCommandBuffer command_buffer, const FrameView& view) const;

    // Anti-aliases the LDR target into the swapchain. Also where the swapchain
    // image first enters COLOR_ATTACHMENT_OPTIMAL, since nothing before this
    // point touches it.
    void record_fxaa(VkCommandBuffer command_buffer, VkImage image, VkImageView image_view,
                     const FrameView& view) const;

    // Draws the selection outline over the scene, reading the id attachment
    // this frame just wrote. Runs before record_pick_copy, which is what fixes
    // each pass's expected source layout to a single value.
    //
    // Runs even with nothing selected: the barrier inside is what leaves the
    // id image in the layout record_pick_copy expects, so skipping the pass
    // would make that layout depend on the selection.
    void record_outline(VkCommandBuffer command_buffer, VkImageView image_view,
                        const FrameView& view, bool has_selection) const;

    // Copies one texel out of the id attachment. Recorded after the scene, so
    // the value read is the one this frame just drew.
    void record_pick_copy(VkCommandBuffer command_buffer, VkOffset2D texel) const;

    // The id copied by the last record_pick_copy. Valid only once the
    // submission carrying that copy has completed; IdBuffer::k_null_id when
    // the pick landed on nothing.
    [[nodiscard]] std::uint32_t read_picked_id() const;

    // Split out of record_scene because the UI draws into the same swapchain
    // image and must get there before it is handed to the presentation engine.
    static void transition_to_present(VkCommandBuffer command_buffer, VkImage image);

    // Resizes every target that follows the swapchain and rewrites the
    // descriptors naming their views. The shadow map is excluded on purpose:
    // its resolution is a quality setting, not a consequence of the window.
    void recreate(VkExtent2D extent);

    [[nodiscard]] std::uint32_t shadow_resolution() const { return shadow_map_.resolution(); }

    // Writes the pipeline cache to disk. Called explicitly at the end of the
    // run rather than from the destructor, so that a crash does not persist a
    // cache built by a run that did not finish.
    void save_pipeline_cache();

private:
    gpu::BindlessSet& bindless_set_;
    const gpu::Swapchain& swapchain_;

    // Per-frame camera, lights and shadow matrix, addressed by device address
    // rather than bound: one slot per frame in flight.
    gpu::Buffer frame_buffer_;
    // One texel of object id, copied out of id_buffer_ on a pick. Host-cached
    // rather than write-combined: this one is read, not written.
    gpu::Buffer pick_buffer_;

    gpu::DepthBuffer depth_buffer_;
    gpu::IdBuffer id_buffer_;
    gpu::HdrTarget hdr_target_;
    gpu::LdrTarget ldr_target_;
    // Not recreated on resize; see recreate().
    gpu::ShadowMap shadow_map_;

    gpu::PipelineCache pipeline_cache_;
    gpu::GraphicsPipeline shadow_pipeline_;
    gpu::GraphicsPipeline scene_pipeline_;
    gpu::GraphicsPipeline outline_pipeline_;
    gpu::GraphicsPipeline tonemap_pipeline_;
    gpu::GraphicsPipeline fxaa_pipeline_;
};

}  // namespace sage::app
