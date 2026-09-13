#include <sage/app/scene_query.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace sage::app {

Bounds compute_scene_bounds(const gpu::SceneGraph& graph) {
    Bounds bounds{glm::vec3(std::numeric_limits<float>::max()),
                  glm::vec3(std::numeric_limits<float>::lowest())};

    for (const gpu::SceneNode& node : graph.nodes()) {
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

LightFit fit_directional_light(const Bounds& bounds, const glm::vec3& light_direction,
                               std::uint32_t shadow_resolution) {
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
    fit.world_texel_size =
        shadow_resolution == 0 ? 0.0F : (extent * 2.0F) / static_cast<float>(shadow_resolution);
    return fit;
}

ChildTable build_child_table(const gpu::SceneGraph& graph) {
    const std::span<const gpu::SceneNode> nodes = graph.nodes();

    ChildTable table;
    table.children.resize(nodes.size());

    for (std::uint32_t i = 0; i < nodes.size(); ++i) {
        if (!nodes[i].alive) {
            continue;
        }
        if (nodes[i].parent.valid()) {
            table.children[nodes[i].parent.index()].push_back(i);
        } else {
            table.roots.push_back(i);
        }
    }
    return table;
}

std::uint32_t collect_lights(const gpu::SceneGraph& graph, std::span<gpu::Light> out) {
    std::uint32_t count = 0;

    for (const gpu::SceneNode& node : graph.nodes()) {
        if (!node.alive || !node.has_light) {
            continue;
        }
        if (count >= out.size()) {
            break;
        }

        gpu::Light& light = out[count];
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

}  // namespace sage::app
