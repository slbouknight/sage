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
#include <sage/gpu/imgui_layer.hpp>
#include <sage/gpu/instance.hpp>
#include <sage/gpu/material_registry.hpp>
#include <sage/gpu/pipeline.hpp>
#include <sage/gpu/pipeline_cache.hpp>
#include <sage/gpu/sampler.hpp>
#include <sage/gpu/scene.hpp>
#include <sage/gpu/surface.hpp>
#include <sage/gpu/swapchain.hpp>
#include <sage/gpu/texture_registry.hpp>
#include <sage/gpu/uploader.hpp>
#include <sage/gpu/window.hpp>

#include <cstdint>
#include <filesystem>
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
    void record_scene(VkCommandBuffer command_buffer, VkImage image, VkImageView image_view,
                      VkExtent2D extent, std::uint32_t frame_slot) const;
    // Split out of record_scene because the UI draws into the same swapchain
    // image and must get there before it is handed to the presentation engine.
    static void transition_to_present(VkCommandBuffer command_buffer, VkImage image);
    // Lays a dockspace over the viewport and, on the first frame, docks the
    // panels into it. Panels place themselves by name from then on, which is
    // what stops a new one landing on top of an existing one.
    void draw_dockspace();
    void draw_ui();
    void draw_hierarchy_panel();

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

    bool dock_layout_built_ = false;

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
    gpu::Buffer frame_buffer_;
    gpu::SceneGraph scene_graph_;
    gpu::Swapchain swapchain_;
    gpu::DepthBuffer depth_buffer_;
    gpu::PipelineCache pipeline_cache_;
    gpu::GraphicsPipeline pipeline_;
    gpu::FramePacer frame_pacer_;
    // Last, so it is destroyed first: its teardown frees Vulkan objects and
    // touches the device, both of which must still be alive.
    gpu::ImGuiLayer imgui_;
};

}  // namespace sage::app
