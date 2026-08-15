#include <sage/core/camera.hpp>

#include <algorithm>
#include <cmath>

namespace sage::core {

namespace {
//  Just under a right angle. At exactly +/- 90 degrees the forward vector
// becomes parallel to world up, the view basis degenerates, and the camera
// flips over. Classic gimbal snap.
constexpr float k_max_pitch = 1.55F;  // ~88.8 degrees

constexpr float k_min_speed = 0.1F;
constexpr float k_max_speed = 100.0F;
constexpr float k_speed_step = 1.15F;

constexpr glm::vec3 k_world_up{0.0F, 1.0F, 0.0F};

}  // namespace

Camera::Camera(glm::vec3 position, float yaw_radians, float pitch_radians)
    : position_(position), yaw_(yaw_radians), pitch_(pitch_radians) {}

glm::vec3 Camera::forward() const {
    return glm::normalize(glm::vec3{std::cos(pitch_) * std::cos(yaw_), std::sin(pitch_),
                                    std::cos(pitch_) * std::sin(yaw_)});
}

void Camera::update(const CameraInput& input, float delta_seconds) {
    yaw_ += input.yaw_delta;
    pitch_ = std::clamp(pitch_ + input.pitch_delta, -k_max_pitch, k_max_pitch);

    const glm::vec3 view_forward = forward();
    const glm::vec3 view_right = glm::normalize(glm::cross(view_forward, k_world_up));

    const float distance = speed_ * delta_seconds;
    position_ += view_forward * input.forward * distance;
    position_ += view_right * input.right * distance;
    // World up, not view up: Unreal's E/Q rise and fall vertically regardless
    // of where the camera is pointing. View-relative would feel weird when pitched.
    position_ += k_world_up * input.up * distance;
}

void Camera::adjust_speed(float scroll_ticks) {
    speed_ = std::clamp(speed_ * std::pow(k_speed_step, scroll_ticks), k_min_speed, k_max_speed);
}

glm::mat4 Camera::view_matrix() const {
    return glm::lookAt(position_, position_ + forward(), k_world_up);
}

}  // namespace sage::core