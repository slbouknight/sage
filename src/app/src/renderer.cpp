#include <sage/app/renderer.hpp>
#include <sage/app/scene_query.hpp>
#include <sage/gpu/frame_pacer.hpp>
#include <sage/gpu/light.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>

namespace sage::app {

namespace {

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

}  // namespace

Renderer::Renderer(gpu::Device& device, gpu::Allocator& allocator, gpu::BindlessSet& bindless_set,
                   const gpu::Swapchain& swapchain)
    : bindless_set_(bindless_set),
      swapchain_(swapchain),
      frame_buffer_(allocator, device, sizeof(FrameData) * gpu::FramePacer::k_frames_in_flight,
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT),
      pick_buffer_(allocator, device, sizeof(std::uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   gpu::BufferAccess::host_read),
      depth_buffer_(allocator, device, swapchain.extent()),
      id_buffer_(allocator, device, swapchain.extent()),
      hdr_target_(allocator, device, swapchain.extent()),
      ldr_target_(allocator, device, swapchain.extent()),
      shadow_map_(allocator, device),
      pipeline_cache_(device, pipeline_cache_path()),
      shadow_pipeline_(device,
                       gpu::GraphicsPipelineDesc{
                           .spirv_path = std::filesystem::path(SAGE_SHADER_DIR) / "shadow.spv",
                           // No colour at all; depth is the entire output.
                           .depth_format = gpu::ShadowMap::format(),
                           .depth_only = true,
                           .depth_bias = true,
                           .set_layout = bindless_set.layout(),
                           .cache = pipeline_cache_.handle(),
                       }),
      scene_pipeline_(device,
                      gpu::GraphicsPipelineDesc{
                          .spirv_path = std::filesystem::path(SAGE_SHADER_DIR) / "mesh.spv",
                          // The HDR target, not the swapchain: this pass no
                          // longer writes anything a display sees directly.
                          .color_format = gpu::HdrTarget::format(),
                          .id_format = gpu::IdBuffer::format(),
                          .depth_format = depth_buffer_.format(),
                          .set_layout = bindless_set.layout(),
                          .cache = pipeline_cache_.handle(),
                      }),
      outline_pipeline_(device,
                        gpu::GraphicsPipelineDesc{
                            .spirv_path = std::filesystem::path(SAGE_SHADER_DIR) / "outline.spv",
                            // Still the swapchain. The outline is an editor
                            // affordance, not a lit surface: drawing it into
                            // the HDR target would put it through the tonemap,
                            // which would quietly change the colour asked for
                            // into a darker, less saturated one.
                            .color_format = swapchain.format(),
                            // No id attachment and no depth: this pass reads
                            // ids, it does not write them, and a full-screen
                            // triangle has nothing to be occluded by.
                            .alpha_blend = true,
                            .cull_backfaces = false,
                            .set_layout = bindless_set.layout(),
                            .cache = pipeline_cache_.handle(),
                        }),
      tonemap_pipeline_(device,
                        gpu::GraphicsPipelineDesc{
                            .spirv_path = std::filesystem::path(SAGE_SHADER_DIR) / "tonemap.spv",
                            // The LDR target, not the swapchain: FXAA sits
                            // between them now.
                            .color_format = gpu::LdrTarget::format(),
                            .cull_backfaces = false,
                            .set_layout = bindless_set.layout(),
                            .cache = pipeline_cache_.handle(),
                        }),
      fxaa_pipeline_(device, gpu::GraphicsPipelineDesc{
                                 .spirv_path = std::filesystem::path(SAGE_SHADER_DIR) / "fxaa.spv",
                                 .color_format = swapchain.format(),
                                 .cull_backfaces = false,
                                 .set_layout = bindless_set.layout(),
                                 .cache = pipeline_cache_.handle(),
                             }) {
    // The descriptors naming these views, written once here and rewritten by
    // recreate(). The shadow map is the exception: it does not follow the
    // swapchain, so its view outlives every resize.
    bindless_set_.write_object_id_image(id_buffer_.view());
    bindless_set_.write_hdr_color_image(hdr_target_.view());
    bindless_set_.write_shadow_map(shadow_map_.view(), shadow_map_.sampler());
    bindless_set_.write_ldr_color_image(ldr_target_.view(), ldr_target_.sampler());
}

void Renderer::recreate(VkExtent2D extent) {
    depth_buffer_.recreate(extent);
    id_buffer_.recreate(extent);
    hdr_target_.recreate(extent);
    ldr_target_.recreate(extent);
    // The old views are gone, so the descriptors naming them have to be
    // rewritten. The shadow map is absent here on purpose: it does not follow
    // the swapchain, so its view is still valid.
    bindless_set_.write_object_id_image(id_buffer_.view());
    bindless_set_.write_hdr_color_image(hdr_target_.view());
    bindless_set_.write_ldr_color_image(ldr_target_.view(), ldr_target_.sampler());
}

void Renderer::save_pipeline_cache() {
    pipeline_cache_.save();
}

std::uint32_t Renderer::read_picked_id() const {
    std::uint32_t id = gpu::IdBuffer::k_null_id;
    pick_buffer_.read(&id, sizeof(id));
    return id;
}

void Renderer::write_frame_data(const FrameView& view) const {
    // One write per frame, into this frame's own slot. Writing a single shared
    // slot would race the GPU, which may still be reading the previous frame's
    // copy. begin_frame() has already waited out the work that used this slot.
    FrameData frame_data;
    frame_data.view_projection = view.view_projection;
    frame_data.camera_position = view.camera_position;

    // Gathered from the graph rather than from members. The shader has always
    // read lights as data; what changed is that the data now comes from nodes,
    // so a light can be placed, parented and dragged like anything else.
    frame_data.light_count = collect_lights(*view.scene, frame_data.lights);
    frame_data.ambient_intensity = view.settings->ambient_intensity;

    // Refitted every frame from live world transforms. A gizmo drag moves
    // geometry, which moves the bounds, which moves the frustum -- so a shadow
    // keeps up with the thing casting it.
    // The shadow map is fitted to one directional light -- the first in the
    // graph, matching the shader, which spends it on the first directional
    // light it evaluates. With none in the scene there is nothing to fit and
    // nothing to cast, so the pass still runs but the lookup is skipped.
    // Which light is the key, and which way it points, is the caller's to
    // decide -- it owns the scene and the notion of "first directional".
    const LightFit fit = fit_directional_light(compute_scene_bounds(*view.scene),
                                               view.key_light_direction, shadow_map_.resolution());
    frame_data.light_view_projection = fit.view_projection;
    // Texels converted to world units here, so the shader stays in world space
    // and the slider keeps meaning the same thing at any scene scale.
    frame_data.shadow_normal_bias = view.settings->shadow_normal_bias_texels * fit.world_texel_size;
    frame_data.shadow_pcf_radius = view.settings->shadow_pcf_radius;
    frame_data.shadow_texel_size = 1.0F / static_cast<float>(shadow_map_.resolution());
    frame_data.shadow_enabled = (view.settings->shadows_enabled && view.has_key_light) ? 1U : 0U;

    frame_buffer_.write(&frame_data, sizeof(frame_data),
                        VkDeviceSize{view.frame_slot} * sizeof(FrameData));
}

void Renderer::record_shadow(VkCommandBuffer command_buffer, const FrameView& view) const {
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
    vkCmdSetDepthBias(command_buffer, view.settings->shadow_depth_bias, 0.0F,
                      view.settings->shadow_slope_bias);

    const VkDeviceAddress frame_address =
        frame_buffer_.device_address() + (VkDeviceSize{view.frame_slot} * sizeof(FrameData));

    for (const gpu::SceneNode& node : view.scene->nodes()) {
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
        vkCmdBindIndexBuffer(command_buffer, view.geometry->buffer(), node.mesh.index_offset,
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

void Renderer::record_scene(VkCommandBuffer command_buffer, const FrameView& view) const {
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

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, scene_pipeline_.handle());

    const VkDescriptorSet descriptor_set = bindless_set_.handle();
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            scene_pipeline_.layout(), 0, 1, &descriptor_set, 0, nullptr);

    // The dockspace's central node, not the whole image: the panels are opaque,
    // so drawing behind them costs fill rate and, worse, makes the projection
    // describe a view wider than the one actually on screen.
    VkViewport viewport{};
    viewport.x = static_cast<float>(view.viewport.offset.x);
    viewport.y = static_cast<float>(view.viewport.offset.y);
    viewport.width = static_cast<float>(view.viewport.extent.width);
    viewport.height = static_cast<float>(view.viewport.extent.height);
    viewport.minDepth = 0.0F;
    viewport.maxDepth = 1.0F;
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);

    vkCmdSetScissor(command_buffer, 0, 1, &view.viewport);

    // Already written by write_frame_data, before the shadow pass -- both
    // passes read the same slot, and the shadow pass needs the light matrix
    // that is in it.
    const VkDeviceAddress frame_address =
        frame_buffer_.device_address() + (VkDeviceSize{view.frame_slot} * sizeof(FrameData));

    for (std::uint32_t index = 0; index < view.scene->nodes().size(); ++index) {
        const gpu::SceneNode& node = view.scene->nodes()[index];
        if (!node.alive || !node.has_mesh) {
            // Deleted, or a pure transform node: glTF hierarchy nodes, and the
            // per-load root.
            continue;
        }
        if (node.editor_only && !view.show_editor_meshes) {
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
        vkCmdPushConstants(command_buffer, scene_pipeline_.layout(), VK_SHADER_STAGE_ALL, 0,
                           sizeof(push), &push);
        vkCmdBindIndexBuffer(command_buffer, view.geometry->buffer(), node.mesh.index_offset,
                             VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(command_buffer, node.mesh.index_count, 1, 0, 0, 0);
    }

    vkCmdEndRendering(command_buffer);
}

void Renderer::record_tonemap(VkCommandBuffer command_buffer, const FrameView& view) const {
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

    // The whole image, not view.viewport: the region behind the panels was
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
    push.exposure = std::exp2(view.settings->exposure_stops);
    push.tonemap_operator = view.settings->tonemap_operator;
    vkCmdPushConstants(command_buffer, tonemap_pipeline_.layout(), VK_SHADER_STAGE_ALL, 0,
                       sizeof(push), &push);

    vkCmdDraw(command_buffer, 3, 1, 0, 0);
    vkCmdEndRendering(command_buffer);
}

void Renderer::record_fxaa(VkCommandBuffer command_buffer, VkImage image, VkImageView image_view,
                           const FrameView& view) const {
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
    push.edge_threshold = view.settings->fxaa_edge_threshold;
    push.edge_threshold_min = view.settings->fxaa_edge_threshold_min;
    push.subpixel_quality = view.settings->fxaa_subpixel_quality;
    push.enabled = view.settings->fxaa_enabled ? 1 : 0;
    vkCmdPushConstants(command_buffer, fxaa_pipeline_.layout(), VK_SHADER_STAGE_ALL, 0,
                       sizeof(push), &push);

    vkCmdDraw(command_buffer, 3, 1, 0, 0);
    vkCmdEndRendering(command_buffer);
}

void Renderer::record_outline(VkCommandBuffer command_buffer, VkImageView image_view,
                              const FrameView& view, bool has_selection) const {
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

    if (!has_selection) {
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
    rendering.renderArea.offset = view.viewport.offset;
    rendering.renderArea.extent = view.viewport.extent;
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &color_attachment;

    vkCmdBeginRendering(command_buffer, &rendering);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, outline_pipeline_.handle());

    const VkDescriptorSet descriptor_set = bindless_set_.handle();
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            outline_pipeline_.layout(), 0, 1, &descriptor_set, 0, nullptr);

    VkViewport viewport{};
    viewport.x = static_cast<float>(view.viewport.offset.x);
    viewport.y = static_cast<float>(view.viewport.offset.y);
    viewport.width = static_cast<float>(view.viewport.extent.width);
    viewport.height = static_cast<float>(view.viewport.extent.height);
    viewport.minDepth = 0.0F;
    viewport.maxDepth = 1.0F;
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(command_buffer, 0, 1, &view.viewport);

    OutlinePushConstants push{};
    push.viewport_origin = {view.viewport.offset.x, view.viewport.offset.y};
    push.viewport_size = {static_cast<std::int32_t>(view.viewport.extent.width),
                          static_cast<std::int32_t>(view.viewport.extent.height)};
    vkCmdPushConstants(command_buffer, outline_pipeline_.layout(), VK_SHADER_STAGE_ALL, 0,
                       sizeof(push), &push);

    // Three vertices, no buffers: the vertex shader builds a full-screen
    // triangle from SV_VertexID.
    vkCmdDraw(command_buffer, 3, 1, 0, 0);
    vkCmdEndRendering(command_buffer);
}

void Renderer::record_pick_copy(VkCommandBuffer command_buffer, VkOffset2D texel) const {
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
    region.imageOffset = {texel.x, texel.y, 0};
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

void Renderer::transition_to_present(VkCommandBuffer command_buffer, VkImage image) {
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

}  // namespace sage::app
