#pragma once

#include <sage/core/math.hpp>

#include <vulkan/vulkan.h>

#include <optional>

namespace sage::app {

// Converts window coordinates into the 3D view's own coordinate systems.
//
// Three systems are in play and they are easy to confuse: ImGui reports
// positions in logical units relative to the main viewport; the attachments
// are framebuffer pixels, which differ from logical units under display
// scaling; and the 3D view is a sub-rect of the framebuffer, because the
// dockspace's panels take the rest. Picking, the context menu and object
// placement all need the same chain, and each used to spell it out.
//
// A plain struct filled in by the caller rather than something that reads
// ImGui itself: that is what lets the arithmetic be tested without a context.
struct ViewportMapping {
    // ImGui's main viewport position and size, in logical units.
    glm::vec2 logical_pos{0.0F};
    glm::vec2 logical_size{0.0F};
    // The swapchain's size, in pixels.
    VkExtent2D framebuffer{};
    // The 3D view within that framebuffer -- the dockspace's central node.
    VkRect2D rect{};

    // False when anything would divide by zero. Every conversion below returns
    // nothing in that case rather than producing a number.
    [[nodiscard]] bool valid() const;

    // A logical window position as a framebuffer pixel position. Not clipped
    // to the 3D view; texel_at is the one that rejects a miss.
    [[nodiscard]] std::optional<glm::vec2> to_framebuffer(glm::vec2 window) const;

    // The framebuffer texel under a window position, or nothing when it falls
    // outside the 3D view. Outside is not a miss, it is not a click in the
    // view at all -- clicking the dockspace border should leave the selection
    // alone, and should not offer to add anything either.
    [[nodiscard]] std::optional<VkOffset2D> texel_at(glm::vec2 window) const;

    // The same position in the 3D view's normalised device coordinates, or
    // nothing when it falls outside.
    //
    // No Y negation: the projection this feeds is already Vulkan's, whose NDC
    // Y runs down the screen exactly as framebuffer rows do.
    [[nodiscard]] std::optional<glm::vec2> ndc_at(glm::vec2 window) const;
};

// The unit direction from `origin` through an NDC point on the far plane.
//
// Unprojecting the far plane alone is enough: the near point is the camera
// position, which is already known exactly. Nothing when the unprojection is
// degenerate.
[[nodiscard]] std::optional<glm::vec3> ray_direction(const glm::mat4& inverse_view_projection,
                                                     const glm::vec3& origin, glm::vec2 ndc);

// Where a ray meets the y == 0 plane, or nothing when it runs along the plane,
// points away from it, or meets it further off than `max_distance`.
//
// A ray with no useful answer here is the common case, not an error: it is
// every shot taken while the camera looks up at the horizon.
[[nodiscard]] std::optional<glm::vec3> ground_plane_hit(const glm::vec3& origin,
                                                        const glm::vec3& direction,
                                                        float max_distance);

}  // namespace sage::app
