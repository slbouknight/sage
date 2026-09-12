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

}  // namespace sage::core