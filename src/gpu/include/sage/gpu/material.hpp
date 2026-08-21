#pragma once

#include <sage/core/math.hpp>

#include <cstddef>
#include <cstdint>

namespace sage::gpu {
// Must match Material in shaders/mesh.slang. Read out of a StructuredBuffer,
// so this is std430: the vec4 forces 16-byte alignment and the trailing three
// scalars are absorbed into the struct's own padding, which is why metallic and
// roughness are free to carry even though nothing reads them yet.
//
// glm's vec4 has alignment 4, not 16, so the alginas is what
// makes the C++ offsets agree with the shader's. Verified against compiled
// SPIR-V: ArrayStride 32, member offsets 0, 16, 20 and 24.
struct Material {
    alignas(16) glm::vec4 base_color_factor{1.0F, 1.0F, 1.0F, 1.0F};
    // Index into the bindless sampled-image array, not a glTF texture index.
    std::uint32_t base_color_texture = 0;
    float metallic = 1.0F;
    float roughness = 1.0F;
};
static_assert(sizeof(Material) == 32);
static_assert(offsetof(Material, base_color_factor) == 0);
static_assert(offsetof(Material, base_color_texture) == 16);
static_assert(offsetof(Material, metallic) == 20);
static_assert(offsetof(Material, roughness) == 24);

}  // namespace sage::gpu