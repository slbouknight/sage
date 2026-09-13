#pragma once

#include <sage/core/math.hpp>
#include <sage/gpu/light.hpp>
#include <sage/gpu/scene.hpp>

#include <cstdint>
#include <span>

namespace sage::app {

// Read-only derivations from a scene graph.
//
// All of these were members of Application, which meant they could only run
// with a Vulkan device, a swapchain and an ImGui context alive -- so none of
// them were tested, despite being the arithmetic most likely to be subtly
// wrong: a bounding box under rotation, an orthographic frustum fitted to it,
// and a light direction pulled out of a transform.
//
// Free functions over `const SceneGraph&` instead. Nothing here touches the
// GPU or the editor's state.

// A world-space axis-aligned bounding box.
struct Bounds {
    glm::vec3 min{0.0F};
    glm::vec3 max{0.0F};
    // True when no geometry contributed. Detected by the box being inverted,
    // which is the state compute_scene_bounds starts from -- a zero-size box
    // at the origin is a legitimate result for a single point and must not be
    // confused with "nothing".
    [[nodiscard]] bool empty() const { return min.x > max.x; }
};

// The box enclosing every live, non-editor mesh, using current world
// transforms. Recomputed rather than remembered, so moving a node with the
// gizmo moves the bounds with it.
//
// Requires SceneGraph::update_transforms() to have run: world transforms are
// read, not derived here.
[[nodiscard]] Bounds compute_scene_bounds(const gpu::SceneGraph& graph);

struct LightFit {
    glm::mat4 view_projection{1.0F};
    // How much world space one shadow-map texel covers. The unit the
    // normal-offset bias is expressed in, so that it means the same thing
    // whatever the scene's scale.
    float world_texel_size = 0.0F;
};

// An orthographic light-space matrix fitted to `bounds`. A directional light
// has no position, so the frustum is placed by the scene rather than by the
// light: centred on the bounds and pulled back along the light direction far
// enough to enclose them.
//
// `shadow_resolution` is the map's edge length in texels, and only affects
// world_texel_size. Empty bounds give a default-constructed fit.
[[nodiscard]] LightFit fit_directional_light(const Bounds& bounds, const glm::vec3& light_direction,
                                             std::uint32_t shadow_resolution);

// Fills `out` from the graph's live light nodes, returning how many were
// written. Stops at the span's size: the frame buffer's light array is a fixed
// size the shader also declares, so the limit is a layout fact rather than a
// policy this function can bend.
//
// Position and direction come from each node's world transform, which is what
// makes the gizmo work on a light at all.
[[nodiscard]] std::uint32_t collect_lights(const gpu::SceneGraph& graph, std::span<gpu::Light> out);

}  // namespace sage::app
