#pragma once

#include <sage/core/math.hpp>

#include <cstddef>
#include <cstdint>

namespace sage::gpu {

enum class LightType : std::uint32_t {
    directional = 0,
    point = 1,
};

// Must match Light in shaders/mesh.slang
//
// Every vec3 is followed by the scalar that fills exactly the padding std430
// would otherwise insert.

struct Light {
    // Point lights only.
    glm::vec3 position{0.0F};
    // Point lights only. Distance at which influence reaches zero.
    // 0 disables the window and leaves pure inverse-square falloff.
    float range = 0.0F;
    // Directional lights only. Points from the light towards the scene,
    // so the shader negates it to get a surface-to-light vector.
    glm::vec3 direction{0.0F, -1.0F, 0.0F};
    float intensity = 1.0F;
    glm::vec3 color{1.0F, 1.0F, 1.0F};
    LightType type = LightType::directional;
};
static_assert(sizeof(Light) == 48);
static_assert(offsetof(Light, position) == 0);
static_assert(offsetof(Light, range) == 12);
static_assert(offsetof(Light, direction) == 16);
static_assert(offsetof(Light, intensity) == 28);
static_assert(offsetof(Light, color) == 32);
static_assert(offsetof(Light, type) == 44);

constexpr std::uint32_t k_max_lights = 8;

}  // namespace sage::gpu