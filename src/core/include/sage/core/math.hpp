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

}  // namespace sage::core