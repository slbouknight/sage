#pragma once

#include <sage/core/math.hpp>

namespace sage::core {

// Per-frame movement intent, already decoded from whatever input device produced it.
// Axis values are expected in [-1, 1]
struct CameraInput {
    float forward = 0.0F;
    float right = 0.0F;
    float up = 0.0F;
    float yaw_delta = 0.0F;    // radians
    float pitch_delta = 0.0F;  // radians
};

// Unreal-style fly camera: yaw/pitch orientation with no roll, movement along
// the view basis except for up/down which stay world-aligned.
//
// Deliberately knows nothing about windows or input devices. It consumes
// a CameraInput and a delta time, which is what makes it testable.
class Camera {
public:
    Camera(glm::vec3 position, float yaw_radians, float pitch_radians);

    void update(const CameraInput& input, float delta_seconds);

    // Multiplies the base movement speed. Clamped to a reasonable range.
    void adjust_speed(float scroll_ticks);

    [[nodiscard]] glm::mat4 view_matrix() const;

    [[nodiscard]] glm::vec3 position() const { return position_; }
    [[nodiscard]] glm::vec3 forward() const;
    [[nodiscard]] float yaw() const { return yaw_; }
    [[nodiscard]] float pitch() const { return pitch_; }
    [[nodiscard]] float speed() const { return speed_; }

private:
    glm::vec3 position_{0.0F};
    float yaw_ = 0.0F;
    float pitch_ = 0.0F;
    float speed_ = 3.0F;  // world units per second
};

}  // namespace sage::core