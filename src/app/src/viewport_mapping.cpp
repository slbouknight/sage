#include <sage/app/viewport_mapping.hpp>

#include <cmath>
#include <cstdint>

namespace sage::app {

bool ViewportMapping::valid() const {
    return logical_size.x > 0.0F && logical_size.y > 0.0F && rect.extent.width > 0 &&
           rect.extent.height > 0;
}

std::optional<glm::vec2> ViewportMapping::to_framebuffer(glm::vec2 window) const {
    if (!valid()) {
        return std::nullopt;
    }
    return glm::vec2{
        (window.x - logical_pos.x) * (static_cast<float>(framebuffer.width) / logical_size.x),
        (window.y - logical_pos.y) * (static_cast<float>(framebuffer.height) / logical_size.y)};
}

std::optional<VkOffset2D> ViewportMapping::texel_at(glm::vec2 window) const {
    const std::optional<glm::vec2> pixel = to_framebuffer(window);
    if (!pixel.has_value()) {
        return std::nullopt;
    }

    // Truncating rather than rounding: a texel covers [n, n+1), so the pixel
    // a coordinate lands in is its floor. Negative coordinates are rejected by
    // the bounds test below before the sign of the truncation could matter.
    const auto texel_x = static_cast<std::int32_t>(pixel->x);
    const auto texel_y = static_cast<std::int32_t>(pixel->y);

    if (texel_x < rect.offset.x || texel_y < rect.offset.y ||
        texel_x >= rect.offset.x + static_cast<std::int32_t>(rect.extent.width) ||
        texel_y >= rect.offset.y + static_cast<std::int32_t>(rect.extent.height)) {
        return std::nullopt;
    }
    return VkOffset2D{texel_x, texel_y};
}

std::optional<glm::vec2> ViewportMapping::ndc_at(glm::vec2 window) const {
    // Routed through texel_at rather than converting independently, so that a
    // position inside the view for picking cannot be outside it for placement.
    // Those were separate conversions once, and they disagreed about whether
    // the viewport's own offset counted.
    if (!texel_at(window).has_value()) {
        return std::nullopt;
    }
    const glm::vec2 pixel = *to_framebuffer(window);

    return glm::vec2{(2.0F * (pixel.x - static_cast<float>(rect.offset.x)) /
                      static_cast<float>(rect.extent.width)) -
                         1.0F,
                     (2.0F * (pixel.y - static_cast<float>(rect.offset.y)) /
                      static_cast<float>(rect.extent.height)) -
                         1.0F};
}

std::optional<glm::vec3> ray_direction(const glm::mat4& inverse_view_projection,
                                       const glm::vec3& origin, glm::vec2 ndc) {
    const glm::vec4 far_point = inverse_view_projection * glm::vec4(ndc.x, ndc.y, 1.0F, 1.0F);
    if (std::abs(far_point.w) < 1e-6F) {
        return std::nullopt;
    }
    const glm::vec3 to_far = glm::vec3(far_point) / far_point.w - origin;
    const float length = glm::length(to_far);
    if (length < 1e-6F) {
        return std::nullopt;
    }
    return to_far / length;
}

std::optional<glm::vec3> ground_plane_hit(const glm::vec3& origin, const glm::vec3& direction,
                                          float max_distance) {
    if (std::abs(direction.y) <= 1e-4F) {
        return std::nullopt;
    }
    const float distance = -origin.y / direction.y;
    if (distance <= 0.0F || distance >= max_distance) {
        return std::nullopt;
    }
    return origin + (direction * distance);
}

}  // namespace sage::app
