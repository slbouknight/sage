#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace sage::core {

// glm::perspective builds an OpenGL-style projection with +Y up. Vulkan's
// framebuffer Y axis points down, so the second row is negated. Wrapping it
// here means no caller can forget, and the flip lives in exactly one place.
//
// GLM_FORCE_DEPTH_ZERO_TO_ONE handles the other half of the convention gap
// (z range) and is set on the sage_core target -- see src/core/CMakeLists.txt.
[[nodiscard]] inline glm::mat4 perspective_vk(float fov_y_radians, float aspect, float near_plane,
                                              float far_plane) {
    glm::mat4 projection = glm::perspective(fov_y_radians, aspect, near_plane, far_plane);
    projection[1][1] *= -1.0F;
    return projection;
}

// The same projection without the Y flip, i.e. OpenGL's convention where NDC
// +Y points up. Needed by anything that projects to the screen itself rather
// than letting Vulkan's viewport do it -- ImGuizmo assumes GL-style NDC, and
// handing it perspective_vk() puts the gizmo upside down relative to the scene
// it is supposed to be sitting on.
[[nodiscard]] inline glm::mat4 perspective_gl(float fov_y_radians, float aspect, float near_plane,
                                              float far_plane) {
    return glm::perspective(fov_y_radians, aspect, near_plane, far_plane);
}

// Orthographic, with the same Y negation as perspective_vk and for the same
// reason. A directional light has no viewpoint, so its shadow pass projects in
// parallel rather than from a position.
//
// The flip matters more here than it looks. A shadow lookup converts NDC to
// texture coordinates by hand rather than letting the viewport do it, so the
// projection used to *rasterise* the map and the one used to *read* it must
// agree on which way Y points. Flipping here makes both sides Vulkan's
// convention; flipping in neither place would also work, and flipping in one
// puts every shadow upside down in a way that looks almost plausible on a
// symmetric model.
[[nodiscard]] inline glm::mat4 ortho_vk(float left, float right, float bottom, float top,
                                        float near_plane, float far_plane) {
    glm::mat4 projection = glm::ortho(left, right, bottom, top, near_plane, far_plane);
    projection[1][1] *= -1.0F;
    return projection;
}

}  // namespace sage::core